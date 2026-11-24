#!/usr/bin/env python3
"""Time several builds or settings of bench_mlp against each other, interleaved.

Each variant is NAME[@REV][#THREADS][:CXXFLAGS[:BENCH_ARGS]]:
  NAME        label in the results
  @REV        build this git revision (or source directory) instead of the working tree
  #THREADS    its own thread counts, e.g. #1 or #1,2,4 (default: --threads)
  CXXFLAGS    extra compiler flags, e.g. -DML_MIN_MADDS_PER_THREAD=5000
  BENCH_ARGS  extra bench_mlp arguments, e.g. --wgrad rows

Variants with the same revision and flags share one build (build-sweep/<key>).
Every round runs each (variant, config, thread count) once, in a freshly
shuffled order, so slow drifts in clock speed or background load hit all
variants alike. Speedups are ratios of medians to the first cell of each
config (the first variant at its first thread count). With --max-spread,
cells whose spread ((max - min) / median) is higher are rerun, up to --reruns
times, and the attempt with the smallest spread is kept.

After timing, bench_mlp_prof runs --profile-rounds more interleaved rounds for
per-op median self times and the thread counts each op chose. Fingerprints are
compared with benchmarks/results/reference.json.

Example:
  python3 benchmarks/interleave.py --out D6-compare --threads 8 --rounds 15 \\
      --variant 'D3@16b56bb' --variant 'adaptive' --variant 'adaptive-rows::--wgrad rows'
  python3 benchmarks/interleave.py --out steps --rounds 0 --profile-rounds 10 --variant 'A@6437532#1' ...
  python3 benchmarks/interleave.py --out final --rounds 10 --max-spread 5 --perf \\
      --variant 'A@6437532#1' --variant 'D#1,2,4,8,16' --variant 'D-rows#8::--wgrad rows'
"""

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bench as rb  # noqa: E402

SWEEP_DIR = os.path.join(rb.ROOT, "build-sweep")


def parse_variant(spec):
    parts = spec.split(":", 2) + ["", ""]
    head, flags, args = parts[0], parts[1], parts[2]
    head, _, threads = head.partition("#")
    name, _, rev = head.partition("@")
    return {"name": name, "rev": rev or None, "cxxflags": flags, "args": shlex.split(args),
            "threads": [int(t) for t in threads.split(",") if t] or None}


def build_variant(v):
    key = hashlib.sha1(f"{v['rev']}|{v['cxxflags']}".encode()).hexdigest()[:10]
    bdir = os.path.join(SWEEP_DIR, key)
    src = rb.ROOT
    if v["rev"] and os.path.isdir(v["rev"]):
        src = v["rev"]
    elif v["rev"]:
        src = os.path.join(bdir, "src")
        if not os.path.exists(src):
            os.makedirs(src)
            archive = subprocess.run(["git", "archive", v["rev"]], cwd=rb.ROOT, check=True, capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", src], input=archive, check=True)
    obj = os.path.join(bdir, "obj")
    if not os.path.exists(os.path.join(obj, "CMakeCache.txt")):
        subprocess.run(["cmake", "-S", src, "-B", obj, "-DCMAKE_BUILD_TYPE=Release",
                        f"-DCMAKE_CXX_FLAGS={v['cxxflags']}"], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", obj, "--target", "bench_mlp", "bench_mlp_prof", "-j"],
                   check=True, capture_output=True)
    v["exe"] = os.path.join(obj, "bench_mlp")
    v["exe_prof"] = os.path.join(obj, "bench_mlp_prof")


PIN_CPU = []   # [cpu] from --pin-single-cpu


def bench(exe, v, cfg, threads):
    # Single-threaded runs can be pinned with taskset. OpenMP builds already bind
    # their first thread to the first core (OMP_PROC_BIND=close); builds from
    # before OpenMP otherwise go wherever the scheduler puts them, and on this
    # laptop only core 0 may boost, so their clock (3.1 vs 4.1 GHz) depends on
    # placement.
    pin = ["taskset", "-c", str(PIN_CPU[0])] if PIN_CPU and threads == 1 else []
    out = subprocess.run(pin + [exe] + rb.bench_args(cfg) + v["args"], cwd=rb.ROOT, check=True, text=True,
                         capture_output=True, env=rb.thread_env(threads)).stdout
    return json.loads(out)


def cpu_max_freqs():
    """scaling_max_freq (kHz) of every CPU, or {} where cpufreq isn't available."""
    freqs = {}
    for name in sorted(os.listdir("/sys/devices/system/cpu"), key=lambda n: (len(n), n)):
        path = f"/sys/devices/system/cpu/{name}/cpufreq/scaling_max_freq"
        if re.fullmatch(r"cpu\d+", name) and os.path.exists(path):
            with open(path) as f:
                freqs[name] = int(f.read())
    return freqs


def spread(ts):
    return 100 * (max(ts) - min(ts)) / statistics.median(ts)


def run_rounds(jobs, variants, rounds, warmup, rng, times, fingerprints, label):
    """Runs every job once per round in shuffled order; appends to times[job]."""
    for rnd in range(warmup + rounds):
        order = jobs[:]
        rng.shuffle(order)
        for j in order:
            vi, n, t = j
            r = bench(variants[vi]["exe"], variants[vi], rb.CONFIGS[n], t)
            # Builds from before OpenMP (parts A-C) don't report threads and run on one.
            if r.get("threads", 1) != t:
                sys.exit(f"{variants[vi]['name']} ran with {r.get('threads', 1)} threads, expected {t}")
            if rnd >= warmup:
                times[j].append(r["train_seconds"])
            fingerprints.setdefault(j, set()).add(r["fingerprint"]["test_pred_hash"])
            fingerprints.setdefault(("fp", j), r["fingerprint"])
        if rnd >= warmup:
            print(f"{label} round {rnd - warmup + 1}/{rounds} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True, help="results name: benchmarks/results/<out>.json")
    ap.add_argument("--configs", default="small,large")
    ap.add_argument("--threads", default="8", help="thread counts for variants without #THREADS")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=1, help="discarded rounds")
    ap.add_argument("--max-spread", type=float, default=None,
                    help="rerun cells whose spread ((max-min)/median, %%) is above this")
    ap.add_argument("--reruns", type=int, default=3, help="at most this many reruns per cell (with --max-spread)")
    ap.add_argument("--profile-rounds", type=int, default=5, help="interleaved bench_mlp_prof rounds for per-op times")
    ap.add_argument("--perf", action="store_true", help="one perf stat run per cell: clock speed during training")
    ap.add_argument("--pin-single-cpu", type=int, default=None,
                    help="run single-threaded cells (timed and profiled) on this CPU with taskset")
    ap.add_argument("--allow-uneven-clocks", action="store_true",
                    help="run even if CPUs have different frequency caps (scaling_max_freq)")
    ap.add_argument("--seed", type=int, default=1, help="shuffle seed")
    args = ap.parse_args()

    # Uneven caps make a run's speed depend on which CPU it lands on (on this
    # laptop, switching power profile once left only CPU 0 able to boost).
    freqs_start = cpu_max_freqs()
    if len(set(freqs_start.values())) > 1 and not args.allow_uneven_clocks:
        sys.exit("CPUs have different scaling_max_freq caps: " + ", ".join(
            f"{c}={f}" for c, f in freqs_start.items()) + " (--allow-uneven-clocks to run anyway)")

    variants = [parse_variant(s) for s in args.variant]
    if args.pin_single_cpu is not None:
        PIN_CPU[:] = [args.pin_single_cpu]
    names = [n for n in args.configs.split(",") if n]
    default_threads = [int(t) for t in args.threads.split(",") if t]
    for v in variants:
        print(f"building {v['name']}", flush=True)
        build_variant(v)

    reference = rb.load_results("reference")
    jobs = [(vi, n, t) for n in names for vi, v in enumerate(variants) for t in (v["threads"] or default_threads)]
    times = {j: [] for j in jobs}
    fingerprints = {}
    rng = random.Random(args.seed)
    # --rounds 0: profile only (no timing rounds, no reruns).
    if args.rounds:
        run_rounds(jobs, variants, args.rounds, args.warmup, rng, times, fingerprints, "timing")

    # Cells whose spread is too high get rerun (interleaved among themselves);
    # the attempt with the smallest spread is kept and all attempts are saved.
    attempts = {j: [times[j]] for j in jobs}
    if args.max_spread is not None and args.rounds:
        for rerun in range(args.reruns):
            bad = [j for j in jobs if spread(min(attempts[j], key=spread)) > args.max_spread]
            if not bad:
                break
            print(f"rerun {rerun + 1}: {len(bad)} cells above {args.max_spread}% spread", flush=True)
            fresh = {j: [] for j in bad}
            run_rounds(bad, variants, args.rounds, 0, rng, fresh, fingerprints, f"rerun {rerun + 1}")
            for j in bad:
                attempts[j].append(fresh[j])
        for j in jobs:
            times[j] = min(attempts[j], key=spread)

    results = {"label": args.out, "environment": rb.environment(os.path.join(rb.ROOT, "build")),
               "pin_single_cpu": args.pin_single_cpu,
               "rounds": args.rounds, "profile_rounds": args.profile_rounds, "max_spread_percent": args.max_spread,
               "variants": [], "results": {}}
    for v in variants:
        results["variants"].append({k: v[k] for k in ("name", "rev", "cxxflags", "args", "threads")})

    def key(j):
        vi, n, t = j
        return f"{variants[vi]['name']}/{n}/t{t}"

    def reference_status(j):
        fp = fingerprints[("fp", j)]
        if len(fingerprints[j]) > 1:
            return "nondeterministic"
        if reference and j[1] in reference["configs"]:
            return rb.compare_fingerprints(reference["configs"][j[1]]["fingerprint"], fp, 1e-6)[0]
        return "?"

    for j in jobs:
        results["results"][key(j)] = {"variant": variants[j[0]]["name"], "config": j[1], "threads": j[2]}

    for n in (names if args.rounds else []):
        cfg_jobs = [j for j in jobs if j[1] == n]
        base = statistics.median(times[cfg_jobs[0]])
        print(f"\n== {n}: median of {args.rounds} interleaved runs, speedup vs {key(cfg_jobs[0])} ==")
        print(f"  {'variant':<26}{'threads':>8}{'median s':>10}{'spread %':>10}{'speedup':>9}  result")
        for j in cfg_jobs:
            vi, _, t = j
            ts = times[j]
            med = statistics.median(ts)
            q = statistics.quantiles(ts, n=4)
            fp = fingerprints[("fp", j)]
            status = reference_status(j)
            sp = spread(ts)
            flag = "" if args.max_spread is None or sp <= args.max_spread else f" (above {args.max_spread}%)"
            print(f"  {variants[vi]['name']:<26}{t:>8}{med:>10.4f}{sp:>10.1f}{base / med:>8.2f}x  {status}{flag}")
            results["results"][key(j)].update({
                "times": ts, "median_seconds": med, "min_seconds": min(ts), "max_seconds": max(ts),
                "spread_percent": sp, "iqr_percent": 100 * (q[2] - q[0]) / med,
                "attempts": len(attempts[j]), "all_attempts": attempts[j],
                "speedup_vs_first": base / med, "reference": status, "fingerprint": fp,
            })

    # Profiled runs, also interleaved: per-op median self time over the rounds,
    # and which thread counts each op chose (the same in every run).
    # Profile-only runs (--rounds 0) get the warmup round here instead, and
    # their fingerprints are checked against the reference.
    profiles = {j: [] for j in jobs}
    prof_train = {j: [] for j in jobs}
    prof_fps = {}
    warmup = 0 if args.rounds else args.warmup
    for rnd in range(warmup + args.profile_rounds):
        order = jobs[:]
        rng.shuffle(order)
        for j in order:
            vi, n, t = j
            r = bench(variants[vi]["exe_prof"], variants[vi], rb.CONFIGS[n], t)
            if rnd >= warmup:
                profiles[j].append(r.get("profile", []))
                prof_train[j].append(r["train_seconds"])
            prof_fps.setdefault(j, set()).add(r["fingerprint"]["test_pred_hash"])
            prof_fps.setdefault(("fp", j), r["fingerprint"])
        if rnd >= warmup:
            print(f"profile round {rnd - warmup + 1}/{args.profile_rounds} done", flush=True)
    if not args.rounds:
        fingerprints.update(prof_fps)
        for j in jobs:
            results["results"][key(j)]["reference"] = reference_status(j)
            print(f"  {key(j):<32} profiled build: {results['results'][key(j)]['reference']}")
    for j, runs in profiles.items():
        if not runs:
            continue
        ops = []
        for e in runs[0]:
            selfs = [x["self_seconds"] for r in runs for x in r if x["name"] == e["name"]]
            ops.append({"name": e["name"], "calls": e["calls"], "flops": e.get("flops", 0),
                        "median_self_seconds": statistics.median(selfs),
                        "self_seconds": selfs, "threads": e.get("threads", {})})
        results["results"][key(j)]["profile"] = ops
        results["results"][key(j)]["profile_train_seconds"] = prof_train[j]
        results["results"][key(j)]["profile_median_seconds"] = statistics.median(prof_train[j])

    if args.profile_rounds:
        for n in names:
            cfg_jobs = [j for j in jobs if j[1] == n]
            print(f"\n== {n}: per-op median self ms over {args.profile_rounds} profiled runs ==")
            prof = {j: {e["name"]: e for e in results["results"][key(j)]["profile"]} for j in cfg_jobs}
            print(f"  {'op':<24}" + "".join(f"{(variants[j[0]]['name'][:9] + '/' + str(j[2])):>13}" for j in cfg_jobs))
            ops = []
            for j in cfg_jobs:
                ops += [o for o in prof[j] if o not in ops]
            for op in ops:
                cells = ""
                for j in cfg_jobs:
                    e = prof[j].get(op)
                    cells += f"{e['median_self_seconds'] * 1e3:>13.2f}" if e else f"{'-':>13}"
                print(f"  {op[:24]:<24}{cells}")

    # Clock speed and CPU use during training, one perf stat run per cell.
    if args.perf:
        order = jobs[:]
        rng.shuffle(order)
        for j in order:
            vi, n, t = j
            rb.EXTRA_ARGS[:] = variants[vi]["args"]
            try:
                pf = rb.perf_once(variants[vi]["exe"], rb.CONFIGS[n], t)
            except KeyError:
                pf = None   # built before bench_mlp could switch perf on around fit() (D0)
            results["results"][key(j)]["perf"] = pf
        rb.EXTRA_ARGS[:] = []
        for j in jobs:
            pf = results["results"][key(j)]["perf"]
            if pf:
                print(f"  perf {key(j):<32} {pf['ghz']:.2f} GHz, {pf['cpus_utilized']:.2f} CPUs")
            else:
                print(f"  perf {key(j):<32} n/a (no perf control in this build)")

    freqs_end = cpu_max_freqs()
    results["cpu_max_freq_khz"] = {"start": freqs_start, "end": freqs_end}
    if freqs_end != freqs_start:
        print("WARNING: CPU frequency caps changed during the session", file=sys.stderr)
        results["cpu_max_freq_changed"] = True

    out_path = os.path.join(rb.RESULTS_DIR, f"{args.out}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {os.path.relpath(out_path, rb.ROOT)}")


if __name__ == "__main__":
    main()
