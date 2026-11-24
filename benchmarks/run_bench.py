#!/usr/bin/env python3
"""Run the MLPClassifier training benchmark and record the results.

For each config, bench_mlp runs --repeats times (after --warmup discarded
runs) and the median training time is reported. bench_mlp_prof then runs
once for a per-op breakdown. Results go to benchmarks/results/<label>.json.

Correctness checks:
  * every repeat must produce bitwise-identical predictions (determinism);
  * the profiled build must match the plain build;
  * results are compared with benchmarks/results/reference.json (written by
    --save-reference): bitwise identical, within --rtol, or FAIL.

Speedup is the median training time of --compare (default: the reference
run) divided by this run's median.

Threads: --threads 1,2,4,8,16 runs every config at each count (OMP_NUM_THREADS,
with OMP_PROC_BIND=close and OMP_PLACES=cores) and stores it as e.g.
"small/t4". The reference and older single-threaded results are matched by
config name alone. --perf adds a perf stat run per count that counts only
during training, for the average clock speed and CPU utilization.

Examples:
  python3 benchmarks/run_bench.py --label 00-baseline --save-reference
  python3 benchmarks/run_bench.py --label 01-loop-order
  python3 benchmarks/run_bench.py --label 02-x --compare 01-loop-order --configs small
  python3 benchmarks/run_bench.py --label D1-x --compare D0-x --threads 1,2,4,8,16 --perf
"""

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "benchmarks", "results")
REFERENCE = os.path.join(RESULTS_DIR, "reference.json")

# Epoch counts are fixed so every run is comparable with the reference.
CONFIGS = {
    "small": {"hidden": "32,16", "epochs": 100},
    "large": {"hidden": "256,256", "epochs": 3},
}
LR = 0.01
SEED = 42
OMP_ENV = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, **kw)


def build(build_dir):
    if not os.path.exists(os.path.join(build_dir, "CMakeCache.txt")):
        run(["cmake", "-S", ROOT, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", build_dir, "--target", "bench_mlp", "bench_mlp_prof", "-j"])


EXTRA_ARGS = []   # bench_mlp options shared by every run (set from --wgrad)


def bench_args(cfg):
    return ["--hidden", cfg["hidden"], "--epochs", str(cfg["epochs"]), "--lr", str(LR), "--seed", str(SEED)] + EXTRA_ARGS


def thread_env(threads):
    return dict(os.environ, OMP_NUM_THREADS=str(threads), **OMP_ENV)


def bench_once(exe, cfg, threads):
    out = run([exe] + bench_args(cfg), capture_output=True, env=thread_env(threads)).stdout
    return json.loads(out)


def perf_once(exe, cfg, threads):
    """Run under perf stat, counting only during fit(); returns clock and CPU use."""
    with tempfile.TemporaryDirectory() as d:
        ctl, ack, out = (os.path.join(d, n) for n in ("ctl", "ack", "perf.csv"))
        os.mkfifo(ctl)
        os.mkfifo(ack)
        env = dict(thread_env(threads), ML_PERF_CTL=ctl, ML_PERF_ACK=ack, LC_ALL="C")
        r = json.loads(run(["perf", "stat", "-x,", "-e", "task-clock,cycles", "--delay=-1",
                            f"--control=fifo:{ctl},{ack}", "-o", out, "--", exe] + bench_args(cfg),
                           capture_output=True, env=env).stdout)
        counts = {}
        with open(out) as f:
            for line in f:
                fields = line.strip().split(",")
                if len(fields) > 2 and fields[0] and fields[0][0].isdigit():
                    counts[fields[2].split(":")[0]] = float(fields[0])
    task_s = counts["task-clock"] * 1e-3
    return {
        "train_seconds": r["train_seconds"],
        "task_clock_seconds": task_s,
        "cycles": counts["cycles"],
        "ghz": counts["cycles"] / task_s * 1e-9,
        "cpus_utilized": task_s / r["train_seconds"],
    }


def read_cmake_cache(build_dir):
    cache = {}
    try:
        with open(os.path.join(build_dir, "CMakeCache.txt")) as f:
            for line in f:
                m = re.match(r"^([A-Za-z_]+):[A-Z]+=(.*)$", line.strip())
                if m:
                    cache[m.group(1)] = m.group(2)
    except OSError:
        pass
    return cache


def environment(build_dir):
    def capture(cmd):
        try:
            return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True).stdout.strip()
        except OSError:
            return ""

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            m = re.search(r"^model name\s*:\s*(.*)$", f.read(), re.M)
            cpu = m.group(1) if m else cpu
    except OSError:
        pass
    cache = read_cmake_cache(build_dir)
    compiler = cache.get("CMAKE_CXX_COMPILER", "")
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": capture(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(capture(["git", "status", "--porcelain", "--untracked-files=no"])),
        "cpu": cpu,
        "platform": platform.platform(),
        "compiler": capture([compiler, "--version"]).splitlines()[0] if compiler else "",
        "build_type": cache.get("CMAKE_BUILD_TYPE", ""),
        "ml_opt_level": cache.get("ML_OPT_LEVEL", ""),
        "ml_native": cache.get("ML_NATIVE", ""),
        "ml_fma": cache.get("ML_FMA", ""),
        "ml_openmp": cache.get("ML_OPENMP", ""),
        "omp": OMP_ENV,
    }


def rel_diff(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def compare_fingerprints(ref, cur, rtol):
    """Return (status, detail): status is 'identical', 'close' or 'FAIL'."""
    if ref["test_pred_hash"] == cur["test_pred_hash"]:
        return "identical", "bitwise identical predictions"
    worst = 0.0
    for key in ("train_log_loss", "test_log_loss", "test_pred_sum"):
        worst = max(worst, rel_diff(ref[key], cur[key]))
    for a, b in zip(ref["test_pred_head"], cur["test_pred_head"]):
        worst = max(worst, rel_diff(a, b))
    acc_diff = abs(ref["test_accuracy"] - cur["test_accuracy"])
    detail = f"max rel diff {worst:.2e}, accuracy diff {acc_diff:.3f} pp"
    if worst <= rtol and acc_diff <= 0.1:
        return "close", detail
    return "FAIL", detail


def load_results(label):
    path = REFERENCE if label == "reference" else os.path.join(RESULTS_DIR, f"{label}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def same_params(a, b):
    return all(a[k] == b[k] for k in ("hidden", "epochs", "lr", "seed", "train_rows", "features"))


def print_profile(profile, fit_seconds):
    rows = sorted(profile, key=lambda e: e["self_seconds"], reverse=True)
    print(f"    {'op':<28}{'calls':>8}{'self ms':>12}{'total ms':>12}{'self %':>9}{'GFLOP/s':>10}  threads (count:calls)")
    for e in rows:
        pct = 100.0 * e["self_seconds"] / fit_seconds if fit_seconds else 0.0
        # Only kernels that report flops (ML_PROFILE_FLOPS) get a rate.
        flops = e.get("flops", 0)
        rate = f"{flops / e['self_seconds'] * 1e-9:.2f}" if flops and e["self_seconds"] else ""
        threads = " ".join(f"{t}:{c}" for t, c in sorted(e.get("threads", {}).items(), key=lambda x: int(x[0])))
        print(f"    {e['name']:<28}{e['calls']:>8}{e['self_seconds'] * 1e3:>12.2f}"
              f"{e['total_seconds'] * 1e3:>12.2f}{pct:>8.1f}%{rate:>10}  {threads}")


def lookup(results, name, threads):
    """A config's entry in a results file: the same thread count if present,
    else the single-threaded entry stored under the bare name (pre-OpenMP runs)."""
    if results is None:
        return None
    return results["configs"].get(f"{name}/t{threads}") or results["configs"].get(name)


def run_config(args, name, cfg, threads, exe, exe_prof, reference, baseline):
    ok = True
    for _ in range(args.warmup):
        bench_once(exe, cfg, threads)
    runs = []
    for i in range(args.repeats):
        r = bench_once(exe, cfg, threads)
        runs.append(r)
        print(f"  run {i + 1}/{args.repeats}: {r['train_seconds']:.4f} s", flush=True)
    if runs[0]["threads"] != threads:
        sys.exit(f"bench_mlp ran with {runs[0]['threads']} threads, expected {threads} (built without OpenMP?)")

    times = [r["train_seconds"] for r in runs]
    median = statistics.median(times)
    entry = {
        "params": {k: runs[0][k] for k in ("hidden", "epochs", "lr", "seed", "train_rows", "features")},
        "threads": threads,
        "wgrad": runs[0].get("wgrad", "outputs"),
        "times": times,
        "median_seconds": median,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "fingerprint": runs[0]["fingerprint"],
        "checks": {},
    }
    spread = 100.0 * (max(times) - min(times)) / median
    print(f"  median {median:.4f} s  (min {min(times):.4f}, max {max(times):.4f}, spread {spread:.1f}%)"
          f"  = {1e3 * median / cfg['epochs']:.2f} ms/epoch")

    # Correctness: determinism across repeats (at this thread count).
    hashes = {r["fingerprint"]["test_pred_hash"] for r in runs}
    det = len(hashes) == 1
    entry["checks"]["deterministic"] = det
    print(f"  check determinism:  {'PASS' if det else 'FAIL (repeats differ)'}")
    ok &= det

    # Per-op breakdown (and profiled build must compute the same thing).
    if not args.no_profile:
        p = bench_once(exe_prof, cfg, threads)
        entry["profile_train_seconds"] = p["train_seconds"]
        entry["profile"] = p["profile"]
        status, detail = compare_fingerprints(entry["fingerprint"], p["fingerprint"], args.rtol)
        entry["checks"]["profiled_matches"] = status
        print(f"  check profiled build: {'PASS' if status != 'FAIL' else 'FAIL'} ({detail})")
        ok &= status != "FAIL"

    # Correctness against the saved (single-threaded) reference.
    if reference is not None:
        ref_cfg = reference["configs"].get(name)
        if ref_cfg is None or not same_params(ref_cfg["params"], entry["params"]):
            print("  check reference:  SKIP (reference has no matching config)")
        else:
            status, detail = compare_fingerprints(ref_cfg["fingerprint"], entry["fingerprint"], args.rtol)
            entry["checks"]["reference"] = status
            print(f"  check reference:  {'FAIL' if status == 'FAIL' else 'PASS'} ({detail})")
            ok &= status != "FAIL"

    # Speedup against an earlier run.
    if baseline is not None and baseline["label"] != args.label:
        base_cfg = lookup(baseline, name, threads)
        if base_cfg is not None and same_params(base_cfg["params"], entry["params"]):
            speedup = base_cfg["median_seconds"] / median
            base_t = base_cfg.get("threads", 1)
            entry["speedup_vs"] = {"label": baseline["label"], "threads": base_t, "speedup": speedup}
            print(f"  speedup vs {baseline['label']} ({base_t} thread{'s' if base_t > 1 else ''}): {speedup:.2f}x "
                  f"({base_cfg['median_seconds']:.4f} s -> {median:.4f} s)")

    if args.perf:
        entry["perf"] = perf_once(exe, cfg, threads)
        pf = entry["perf"]
        print(f"  perf (training only): {pf['ghz']:.2f} GHz average, {pf['cpus_utilized']:.2f} CPUs utilized")

    if "profile" in entry:
        print(f"  per-op breakdown (bench_mlp_prof, fit {entry['profile_train_seconds']:.4f} s):")
        print_profile(entry["profile"], entry["profile_train_seconds"])

    return entry, ok


def print_scaling(name, entries):
    """Speedup over the lowest thread count, overall and per profiled op."""
    ts = sorted(entries)
    base = entries[ts[0]]
    print(f"\n  {name}: speedup vs {ts[0]} thread{'s' if ts[0] > 1 else ''}")
    head = "".join(f"{f't{t}':>8}" for t in ts)
    print(f"    {'':<28}{head}")
    print(f"    {'train (median)':<28}" + "".join(f"{base['median_seconds'] / entries[t]['median_seconds']:>7.2f}x" for t in ts))
    if all("perf" in entries[t] for t in ts):
        print(f"    {'clock GHz':<28}" + "".join(f"{entries[t]['perf']['ghz']:>8.2f}" for t in ts))
        print(f"    {'CPUs utilized':<28}" + "".join(f"{entries[t]['perf']['cpus_utilized']:>8.2f}" for t in ts))
    if all("profile" in entries[t] for t in ts):
        ops = {e["name"]: e for e in base["profile"]}
        rows = sorted(ops.values(), key=lambda e: e["self_seconds"], reverse=True)
        for e in rows:
            if e["self_seconds"] < 1e-3:
                continue
            cells = ""
            for t in ts:
                other = next((x for x in entries[t]["profile"] if x["name"] == e["name"]), None)
                cells += f"{e['self_seconds'] / other['self_seconds']:>7.2f}x" if other and other["self_seconds"] else f"{'-':>8}"
            print(f"    {e['name']:<28}{cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="name for this run, e.g. 00-baseline")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="discarded runs per config")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated subset of: " + ", ".join(CONFIGS))
    ap.add_argument("--compare", default="reference", help="label of an earlier run to compute speedup against")
    ap.add_argument("--save-reference", action="store_true", help="also store this run as the correctness reference")
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for non-identical fingerprints")
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build"))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-profile", action="store_true", help="skip the bench_mlp_prof run")
    ap.add_argument("--threads", default="1", help="comma-separated OMP_NUM_THREADS values, e.g. 1,2,4,8,16")
    ap.add_argument("--perf", action="store_true", help="also record clock speed during training with perf stat")
    ap.add_argument("--wgrad", choices=["outputs", "rows"], default="outputs",
                    help="how matmul_tn is threaded: split its outputs (identical) or its row sum (close)")
    args = ap.parse_args()
    EXTRA_ARGS[:] = ["--wgrad", args.wgrad]

    if args.label == "reference" or not re.fullmatch(r"[A-Za-z0-9._-]+", args.label):
        ap.error("label must be [A-Za-z0-9._-]+ and not 'reference'")
    names = [n for n in args.configs.split(",") if n]
    threads = [int(t) for t in args.threads.split(",") if t]
    for n in names:
        if n not in CONFIGS:
            ap.error(f"unknown config {n!r}")

    build_dir = os.path.abspath(args.build_dir)
    if not args.no_build:
        build(build_dir)
    exe = os.path.join(build_dir, "bench_mlp")
    exe_prof = os.path.join(build_dir, "bench_mlp_prof")

    reference = None if args.save_reference else load_results("reference")
    baseline = load_results(args.compare)
    if args.compare != "reference" and baseline is None:
        sys.exit(f"no results found for --compare {args.compare}")

    env = environment(build_dir)
    print(f"[{args.label}] commit {env['git_commit']}{' (dirty)' if env['git_dirty'] else ''}, "
          f"{env['compiler']}, {env['build_type']} -O{env['ml_opt_level'] or '?'}"
          f"{' native' if env['ml_native'] == 'ON' else ''}{' fma' if env['ml_fma'] == 'ON' else ''}"
          f", wgrad={args.wgrad}, cpu: {env['cpu']}")
    if reference is None and not args.save_reference:
        print("  note: no reference saved yet; run with --save-reference to enable correctness checks")

    failed = False
    results = {"label": args.label, "environment": env, "configs": {}}
    for name in names:
        cfg = CONFIGS[name]
        entries = {}
        for t in threads:
            key = f"{name}/t{t}"
            print(f"\n== {name}: hidden {cfg['hidden']}, {cfg['epochs']} epochs, {t} thread{'s' if t > 1 else ''} ==")
            entry, ok = run_config(args, name, cfg, t, exe, exe_prof, reference, baseline)
            failed |= not ok
            results["configs"][key] = entries[t] = entry
        if len(threads) > 1:
            print_scaling(name, entries)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {os.path.relpath(out_path, ROOT)}")
    if args.save_reference:
        shutil.copyfile(out_path, REFERENCE)
        print(f"saved {os.path.relpath(REFERENCE, ROOT)} (correctness reference)")

    if failed:
        print("\nCORRECTNESS CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
