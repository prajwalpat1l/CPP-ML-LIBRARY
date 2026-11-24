# MLP benchmarks

The benchmark harness measures `MLPClassifier::fit()` and checks prediction
fingerprints alongside timing. Run commands from the repository root.

## Run a measurement

Requires CMake 3.16+, a C++20 compiler, and Python 3. OpenMP is required for
parallel measurements; serial measurements work without it.

```sh
python3 benchmarks/run_bench.py --label local --configs small --threads 1
python3 benchmarks/run_bench.py --label scaling --configs large --threads 1,2,4,8
```

The runner builds Release drivers in `build/`, performs a warmup and five timed
repeats by default, and reports median training time. It also runs a profiled
build, checks deterministic predictions, and compares results with
`benchmarks/results/reference.json`. New results go to
`benchmarks/results/<label>.json` and are ignored by Git.

See [build instructions](building.md) for selecting serial mode or configuring
OpenMP. Setting a thread count does not enable parallelism in a serial build.

## Configurations

| Name | Hidden layers | Epochs | Training rows | Learning rate | Seed |
| --- | --- | --- | --- | --- | --- |
| `small` | 32, 16 | 100 | 8,000 | 0.01 | 42 |
| `large` | 256, 256 | 3 | 8,000 | 0.01 | 42 |

Both configurations use `data/Diabetes_Prediction.csv`, eight numeric input
features after preprocessing, and a single sigmoid output. See the
[dataset notes](../data/README.md) before using the bundled data elsewhere.

## Interpret results

- Compare runs on the same machine, with the same compiler, flags, and thread
  settings. A saved baseline's timings do not establish speedup on another machine.
- The committed JSON files are historical comparison fixtures, not measurements
  of the current checkout. Some recorded revisions are unavailable in this history.
- The default gradient split preserves summation order. The optional
  `--wgrad rows` path regroups sums and is checked with a tolerance.
- `ML_NATIVE`, `ML_FMA`, and `ML_OPT_LEVEL` change code generation and may affect
  timings or numerical agreement. Default Release builds use optimization level
  2, disable machine-specific instructions, and disable fused multiply-add.

## Other tools

`interleave.py` alternates variants to reduce timing drift. A variant can select
a revision that exists in your checkout or use the current working tree:

```sh
python3 benchmarks/interleave.py --out split-comparison --configs small --threads 1 \
  --variant current --variant 'rows::--wgrad rows'
```

`plot_results.py` produces charts and requires `matplotlib`. Some charts require
additional intermediate result files that are not bundled. The optional `--perf`
mode in the benchmark tools requires Linux `perf`.
