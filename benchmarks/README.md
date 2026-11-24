# Benchmarks

`bench_mlp.cpp` trains the binary MLP and emits JSON timing and prediction
fingerprints. CMake builds it twice: `bench_mlp` for timing and `bench_mlp_prof`
for per-operation profiling.

Run from the repository root:

```sh
python3 benchmarks/run_bench.py --label local --configs small --threads 1
```

- `run_bench.py`: builds the drivers, repeats runs, and checks fingerprints.
- `interleave.py`: compares variants in alternating order to reduce timing bias.
- `plot_results.py`: optional charts; requires `matplotlib`.
- `results/reference.json`: saved baseline measurements and fingerprints.
- `results/final-milestones.json`: saved interleaved milestone measurements.

Generated results and plots are ignored by Git. The committed JSON files are
saved measurements, not newly measured results. Some
charts need additional intermediate result files that are not bundled here.

See [build options](../docs/building.md) and [the performance write-up](../docs/benchmarks.md)
for hardware, compiler flags, reproduction details, and comparison limitations.
