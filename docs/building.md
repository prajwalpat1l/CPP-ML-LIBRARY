# Building and verification

Requires CMake 3.16+ and a C++20 compiler. GCC and Clang are the initial targets.
OpenMP is optional. On macOS, Apple Clang may require a separately installed
OpenMP runtime; a serial build needs no additional runtime.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/linear_regression
```

Run commands from the repository root. Multi-configuration generators additionally
need `--config Release` when building, `-C Release` when testing, and
`./build/Release/linear_regression` when running the example.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `ML_OPENMP` | `ON` | Enable OpenMP if found; otherwise warn and build serially |
| `ML_BUILD_EXAMPLES` | `ON` | Build synthetic regression and dataset-based MLP examples |
| `ML_BUILD_BENCHMARKS` | `ON` | Build `bench_mlp` and `bench_mlp_prof` |
| `BUILD_TESTING` | `ON` | Build and register correctness checks through CTest |
| `ML_OPT_LEVEL` | `2` | GCC/Clang Release optimization level, from 0 to 3 |
| `ML_NATIVE` | `OFF` | Enable machine-specific instructions on GCC/Clang |
| `ML_FMA` | `OFF` | Permit fused multiply-add and tolerance-based kernel checks |

Select serial mode explicitly with:

```sh
cmake -S . -B build-serial -DML_OPENMP=OFF
cmake --build build-serial --parallel
ctest --test-dir build-serial --output-on-failure
```

The test executable exercises numerical kernels, small MLPs, and basic regression
learning. Thread-selection thresholds are checked only in OpenMP builds. A serial
test pass does not verify parallel behavior. The synthetic example is also
registered as a CTest check when examples are enabled.

## Apple Clang with Homebrew OpenMP

If `libomp` is installed through Homebrew but automatic detection fails, configure
with explicit paths (adjust the prefix on Intel Macs):

```sh
cmake -S . -B build-openmp -DML_OPENMP=ON \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
  -DOpenMP_CXX_INCLUDE_DIR=/opt/homebrew/opt/libomp/include
cmake --build build-openmp --parallel
ctest --test-dir build-openmp --output-on-failure
```

## Benchmark tools

```sh
python3 benchmarks/run_bench.py --label local --configs small --threads 1
```

The runner configures `build/` only if it has no CMake cache. To force serial mode,
configure that directory with `-DML_OPENMP=OFF` first. Actual thread scaling needs
an OpenMP build. See [the benchmark notes](benchmarks.md) before comparing runs.
The optional plotting script needs `matplotlib`; `--perf` requires Linux `perf`.
