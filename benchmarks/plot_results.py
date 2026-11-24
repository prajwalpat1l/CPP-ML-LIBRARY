#!/usr/bin/env python3
"""Plot the benchmark results in benchmarks/results/ to benchmarks/plots/ (PNG and SVG).

  speedup_by_stage    training speedup over the original code: A -> B1 -> C3 -> D (8 threads)
  thread_scaling      final version: speedup vs threads, multiplies vs elementwise ops
  gflops_by_step      GFLOP/s of each multiply kernel at every single-threaded step
                      (profile shares × plain-build training time),
                      against the single-core ceiling
  perf_counters       IPC and L1 miss rate (perf stat -d) for B3a, B3b, C1 and C3
  split_cost          split outputs (bit-identical) vs split rows (close) at 8 threads

Milestones, thread scaling and split cost come from the final interleaved set
(--final, default benchmarks/results/final-milestones.json). The per-step GFLOP/s and
the clock speed for their ceiling come from one profiling session of every step
at 1 thread (--steps, default benchmarks/results/final-steps-profile.json): each
kernel's time is its share of the profiled run's self time × the plain build's
training time from that session. Made with:

  python3 benchmarks/interleave.py --out final-steps-profile --rounds 10 --profile-rounds 10 \\
      --pin-single-cpu 0 --perf --variant 'A@6437532#1' ... --variant 'D@1183b21#1'   (every step in STEPS)

Usage: python3 benchmarks/plot_results.py [--final NAME] [--steps NAME]
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "benchmarks", "results")
PLOTS = os.path.join(ROOT, "benchmarks", "plots")

# Reference light palette (dataviz skill): categorical slots in fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL = "#8a8984"

CONFIG_TITLES = {"small": "Small net (32-16-1)", "large": "Large net (256-256-1)"}

# Zen 3 (Ryzen 7 5800H) can issue two 128-bit FP multiplies and two 128-bit FP
# adds per cycle. The default build has no -march=native, so SSE2 (2 doubles
# per vector) and no FMA: 2·2 + 2·2 = 8 double-precision flops per cycle.
FLOPS_PER_CYCLE = 8

# Every step in commit order, with its revision: the variants of the
# single-threaded profiling session (--steps) that gflops_by_step reads. D is
# the final version (D7) at 1 thread.
STEPS = [
    ("A", "6437532"), ("B4", "4ad1cb9"), ("B3a", "9dbd809"), ("B3b", "698cb4a"), ("B2", "c16ef65"),
    ("B1", "a26cfed"), ("C0", "f328f15"), ("C1", "851de94"), ("C2", "79f138c"), ("C3", "58c7179"),
    ("D", "1183b21"),
]

MULTIPLY_OPS = {"Matrix::matmul", "Matrix::matmul_bias", "Matrix::matmul_tn", "Matrix::matmul_nt", "Matrix::matmul_into"}
ELEMENTWISE_OPS = {"MLP::relu", "MLP::sigmoid", "MLP::relu_backward", "MLP::sgd_step", "Matrix::column_sums"}


def load(name):
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "font.size": 9, "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
        "svg.fonttype": "none", "axes.axisbelow": True,
        "svg.hashsalt": "ml-plots",   # fixed SVG element ids, so reruns don't churn the files
    })


def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = os.path.join(PLOTS, f"{name}.{ext}")
        # No date or matplotlib version stamped in: same data, same bytes.
        metadata = {"Date": None, "Creator": None} if ext == "svg" else {"Software": None}
        fig.savefig(path, dpi=150, bbox_inches="tight", metadata=metadata)
        paths.append(os.path.relpath(path, ROOT))
    plt.close(fig)
    print("saved " + ", ".join(paths))


def note(fig, text):
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8, color=INK_2, wrap=True)


def bars(ax, labels, values, color, fmt):
    x = range(len(values))
    ax.bar(x, values, width=0.6, color=color, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(list(x), labels)
    ax.tick_params(axis="x", length=0)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, fmt(v), ha="center", va="bottom", fontsize=9, color=INK)


def final_cell(final, variant, config, threads):
    return final["results"][f"{variant}/{config}/t{threads}"]


def profile_self(cell):
    """op name -> median self seconds, for a final-set cell."""
    return {e["name"]: e["median_self_seconds"] for e in cell.get("profile", [])}


def speedup_by_stage(final):
    stages = [("Original (A)", "A", 1), ("B1", "B1", 1), ("C3", "C3", 1), ("D, 8 threads", "D", 8)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, config in zip(axes, ("small", "large")):
        times = [final_cell(final, v, config, t)["median_seconds"] for _, v, t in stages]
        speedups = [times[0] / x for x in times]
        x = list(range(len(stages)))
        ax.plot(x, speedups, color=BLUE, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        for i, (s, t) in enumerate(zip(speedups, times)):
            # Above-left: the line always arrives from the lower left.
            ax.annotate(f"{s:.1f}x\n{t * 1e3:.0f} ms", (i, s), xytext=(-6, 6), textcoords="offset points",
                        ha="right", va="bottom", fontsize=9, color=INK)
        # Gain from each milestone to the next, in a row between the x positions.
        for i in range(len(stages) - 1):
            ax.text(i + 0.5, 0.72, f"×{speedups[i + 1] / speedups[i]:.1f}", ha="center", va="center",
                    fontsize=9, color=INK_2, fontweight="bold")
        ax.set_xticks(x, [s[0] for s in stages])
        ax.set_xlim(-0.6, len(stages) - 0.6)
        ax.tick_params(axis="x", length=0)
        ax.set_yscale("log")
        ax.set_ylim(0.55, 200)
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.set_title(CONFIG_TITLES[config], loc="left")
    axes[0].set_ylabel("training speedup over original (log scale)")
    fig.suptitle("Training speedup by stage", x=0.01, ha="left", fontsize=13, fontweight="bold")
    note(fig, f"Median of {final['rounds']} interleaved runs per point (benchmarks/results/{final['label']}.json); "
              "point labels: speedup over the original and training time; ×N: gain over the previous milestone. "
              "All stages bit-identical to the original.")
    fig.tight_layout()
    save(fig, "speedup_by_stage")


def thread_scaling(final):
    threads = sorted(c["threads"] for c in final["results"].values() if c["variant"] == "D" and c["config"] == "small")
    cores = 8
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, config in zip(axes, ("small", "large")):
        cells = {t: final_cell(final, "D", config, t) for t in threads}
        groups = {
            "Multiplies": lambda p: sum(v for k, v in p.items() if k in MULTIPLY_OPS),
            "Elementwise ops": lambda p: sum(v for k, v in p.items() if k in ELEMENTWISE_OPS),
        }
        series = []
        for (label, total), color in zip(groups.items(), (BLUE, ORANGE)):
            secs = {t: total(profile_self(cells[t])) for t in threads}
            series.append((label, [secs[threads[0]] / secs[t] for t in threads], color))
        series.append(("Training (total)", [cells[threads[0]]["median_seconds"] / cells[t]["median_seconds"] for t in threads], AQUA))

        ax.axvspan(cores, threads[-1], color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.axvline(cores, color=INK_2, linewidth=1, zorder=1)
        ax.text(cores * 0.94, 13.5, f"{cores} physical cores", ha="right", va="center", fontsize=8, color=INK_2)
        ax.text((cores * threads[-1]) ** 0.5, 13.5, "hyperthreads", ha="center", va="center", fontsize=8, color=INK_2)
        ax.plot(threads, threads, color=NEUTRAL, linewidth=1, linestyle=(0, (4, 3)), label="Ideal (linear)")
        at8 = threads.index(cores)
        # Label the 8-thread values: the highest above its point, the others below.
        top = max(series, key=lambda s: s[1][at8])[0]
        for label, ys, color in series:
            ax.plot(threads, ys, color=color, linewidth=2, marker="o", markersize=5, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
            ax.annotate(f"{ys[at8]:.1f}x", (cores, ys[at8]), xytext=(-6, 9 if label == top else -9),
                        textcoords="offset points", ha="right", va="center", fontsize=8, color=INK,
                        fontweight="bold", zorder=4,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8, 16]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.set_ylim(0.4, 20)
        ax.set_xlim(0.85, threads[-1] * 1.15)
        ax.grid(True, axis="both")
        ax.set_xlabel("threads (OMP_NUM_THREADS)")
        ax.set_title(CONFIG_TITLES[config], loc="left")
    axes[0].set_ylabel("speedup over 1 thread (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Final version: speedup vs threads", x=0.01, ha="left", fontsize=13, fontweight="bold")
    note(fig, "Labels: speedup at 8 threads. Groups sum per-op median self time from the profiled build: multiplies = "
              "matmul_bias, matmul_into (dA), matmul_tn (dW); elementwise = relu, sigmoid, relu_backward, column_sums, "
              "sgd_step. Ryzen 7 5800H: 8 cores, 2 hardware threads each; OMP_PROC_BIND=close, OMP_PLACES=cores.")
    fig.tight_layout()
    save(fig, "thread_scaling")


def kernel_ops(names):
    """Profile names of the forward, dA and dW multiplies at a step (None before B3a split them)."""
    if "Matrix::matmul_tn" not in names:
        return None
    forward = "Matrix::matmul_bias" if "Matrix::matmul_bias" in names else "Matrix::matmul"
    dA = "Matrix::matmul_into" if "Matrix::matmul_into" in names else "Matrix::matmul_nt"
    return {"Forward A·W": forward, "dA = dZ·Wᵀ": dA, "dW = Aᵀ·dZ": "Matrix::matmul_tn"}


def gflops_by_step(steps_run):
    labels = [label for label, _ in STEPS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ghz_used = []
    for ax, config in zip(axes, ("small", "large")):
        # Kernel time = its share of the profiled run's self time × the plain
        # build's median training time from the same session, so a profiled
        # run that happened to run at another clock can't skew the rate.
        selfs = {}
        for label in labels:
            cell = final_cell(steps_run, label, config, 1)
            prof = profile_self(cell)
            scale = cell["median_seconds"] / sum(prof.values())
            selfs[label] = {k: v * scale for k, v in prof.items()}
        # Useful flops (2·m·n·q per multiply) are the same at every step; take
        # them from C0, the first step that counts them.
        c0 = {e["name"]: e["flops"] for e in final_cell(steps_run, "C0", config, 1)["profile"]}
        flops = {"Forward A·W": c0["Matrix::matmul_bias"], "dA = dZ·Wᵀ": c0["Matrix::matmul_nt"],
                 "dW = Aᵀ·dZ": c0["Matrix::matmul_tn"]}
        total_flops = sum(flops.values())

        x = list(range(len(labels)))
        overall = [total_flops / sum(v for k, v in selfs[s].items() if k in MULTIPLY_OPS) * 1e-9 for s in labels]
        for k, color in zip(flops, (BLUE, ORANGE, AQUA)):
            pts = [(i, flops[k] / selfs[s][kernel_ops(selfs[s])[k]] * 1e-9)
                   for i, s in enumerate(labels) if kernel_ops(selfs[s])]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=k)
        ax.plot(x, overall, color=YELLOW, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label="All multiplies")
        # The lines converge from C2 on, so only the combined value is labelled.
        ax.annotate(f"{overall[-1]:.1f}", (x[-1], overall[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)

        ghz = final_cell(steps_run, "D", config, 1)["perf"]["ghz"]
        ghz_used.append(ghz)
        ceiling = FLOPS_PER_CYCLE * ghz
        ax.axhline(ceiling, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(x[-1], ceiling * 1.1, f"single-core ceiling {ceiling:.0f} GFLOP/s", fontsize=8, color=INK_2,
                va="bottom", ha="right")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 5, 10, 20, 50]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(0.3, 60)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.4, len(x) - 0.2)
        ax.tick_params(axis="x", length=0)
        ax.set_title(CONFIG_TITLES[config], loc="left")
    axes[0].set_ylabel("GFLOP/s, one thread (log scale)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Multiply kernels: GFLOP/s at each step (one thread)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    note(fig, f"One interleaved session (benchmarks/results/{steps_run['label']}.json): each kernel's time is its share of "
              f"the profiled build's self time (median of {steps_run['profile_rounds']} runs) × the plain build's median "
              f"training time ({steps_run['rounds']} runs), all at 1 thread on CPU {steps_run.get('pin_single_cpu')}. "
              f"Ceiling assumes Zen 3 issues two 128-bit SSE2 multiplies "
              f"and two 128-bit adds per cycle with no FMA (the default build has no -march=native): "
              f"{FLOPS_PER_CYCLE} double-precision flops/cycle × the clock perf measured during single-threaded "
              f"training in that session ({min(ghz_used):.2f}-{max(ghz_used):.2f} GHz). Flops are the useful 2·m·n·q "
              "per multiply at every step. A and B4 had one matmul for all three products (only 'All multiplies' "
              "shown); the label is the combined value at D.")
    fig.tight_layout()
    save(fig, "gflops_by_step")


def perf_counters():
    steps = ["B3a", "B3b", "C1", "C3"]
    ipc, miss = [], []
    for s in steps:
        with open(os.path.join(RESULTS, "perf", f"{s}.txt")) as f:
            text = f.read()
        count = lambda event: float(re.search(r"^\s*([\d.]+)\s+" + re.escape(event), text, re.M).group(1))
        ipc.append(count("instructions:u") / count("cpu-cycles:u"))
        miss.append(float(re.search(r"L1-dcache-load-misses:u\s+#\s+([\d.]+) %", text).group(1)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    bars(axes[0], steps, ipc, BLUE, lambda v: f"{v:.2f}")
    axes[0].set_title("Instructions per cycle", loc="left")
    axes[0].set_ylim(0, max(ipc) * 1.2)
    bars(axes[1], steps, miss, BLUE, lambda v: f"{v:.1f}%")
    axes[1].set_title("L1 data-cache load miss rate", loc="left")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    axes[1].set_ylim(0, max(miss) * 1.2)
    fig.suptitle("Large net, one thread: perf stat -d", x=0.01, ha="left", fontsize=13, fontweight="bold")
    note(fig, "Whole bench_mlp process (loading included), 3 epochs of 256-256-1; benchmarks/results/perf/*.txt. "
              "B3a: transpose-free (old loop order), B3b: streaming row order, C1: i-k-j forward, C3: 4-row blocking. "
              "Miss rate is L1-dcache-load-misses / L1-dcache-loads.")
    fig.tight_layout()
    save(fig, "perf_counters")


def split_cost(final):
    configs = ("small", "large")
    outputs = [final_cell(final, "D", c, 8)["median_seconds"] * 1e3 for c in configs]
    rows = [final_cell(final, "D-rows", c, 8)["median_seconds"] * 1e3 for c in configs]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.34
    for i, c in enumerate(configs):
        for dx, v, color in ((-w / 2 - 0.01, outputs[i], BLUE), (w / 2 + 0.01, rows[i], ORANGE)):
            ax.bar(i + dx, v, width=w, color=color, edgecolor=SURFACE, linewidth=2)
            ax.text(i + dx, v + max(outputs) * 0.02, f"{v:.0f} ms", ha="center", va="bottom", fontsize=9)
        ax.text(i, max(outputs[i], rows[i]) + max(outputs) * 0.12,
                f"rows {outputs[i] / rows[i]:.2f}x faster", ha="center", fontsize=9, color=INK_2)
    ax.legend(handles=[Patch(color=BLUE, label="Split outputs (default, bit-identical)"),
                       Patch(color=ORANGE, label="Split rows (opt-in, close)")], loc="upper left")
    ax.set_xticks(range(len(configs)), [CONFIG_TITLES[c] for c in configs])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(outputs) * 1.4)
    ax.set_ylabel("training time, 8 threads (ms)")
    ax.set_title("Cost of staying bit-identical: weight-gradient split", loc="left", fontsize=13)
    note(fig, f"Median of {final['rounds']} interleaved runs (benchmarks/results/{final['label']}.json). Split rows sums "
              "per-thread partials: deterministic for a given shape and thread count, not bit-identical.")
    fig.tight_layout()
    save(fig, "split_cost")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", default="final-milestones", help="interleaved results file with the milestones")
    ap.add_argument("--steps", default="final-steps-profile", help="profile-only session of every step at 1 thread")
    args = ap.parse_args()
    style()
    final = load(args.final)
    speedup_by_stage(final)
    thread_scaling(final)
    gflops_by_step(load(args.steps))
    perf_counters()
    split_cost(final)


if __name__ == "__main__":
    main()
