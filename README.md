# CPP-ML-LIBRARY

A small, header-only C++20 machine learning library for learning how models work,
experimenting with numerical code, and measuring training performance.

The library includes matrix operations, tabular data utilities, linear and logistic
regression, a binary decision tree, and a binary multilayer perceptron (MLP).
It is an early-stage educational project with reference-based kernel tests and an
MLP benchmark harness.

## Get started

You need **CMake 3.16+** and a **C++20 compiler**. OpenMP is optional; the build
falls back to serial execution when it is unavailable. Python 3 is only needed
for the benchmark scripts.

```sh
git clone https://github.com/prajwalpat1l/CPP-ML-LIBRARY.git
cd CPP-ML-LIBRARY
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/linear_regression
```

The example learns `y = 1 + 2x` and predicts approximately `7` for `x = 3`.
It needs no downloaded data. For a serial build, add `-DML_OPENMP=OFF` to the
configure command. With a multi-configuration generator, build and test with
`--config Release` and `-C Release`, respectively; executables are under
`build/Release/`.

## Use the library

Include the header for the component you need:

```cpp
#include "cpp_ml/models/LinearRegression.hpp"

Matrix X(3, 1), y(3, 1);
for (size_t i = 0; i < 3; ++i) {
    X(i, 0) = static_cast<double>(i);
    y(i, 0) = 1.0 + 2.0 * X(i, 0);
}

LinearRegression model;
model.fit(X, y, 0.1, 1000); // learning rate, epochs
Matrix predictions = model.predict(X);
```

See [the complete example](examples/linear_regression.cpp). Model and matrix
classes currently live in the global namespace; `cpp_ml` is the include prefix.

For a CMake application, put this repository under `external/cpp-ml-library`:

```cmake
set(ML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ML_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(external/cpp-ml-library)
target_link_libraries(your_app PRIVATE cpp_ml::cpp_ml)
```

You can also compile directly with `-std=c++20 -I/path/to/CPP-ML-LIBRARY/include`.

## Components

| Component | Header | Purpose |
| --- | --- | --- |
| `Matrix` | `cpp_ml/core/Matrix.hpp` | Dense double-precision storage, arithmetic, and training kernels |
| `DataFrame`, `CSVReader` | `cpp_ml/data/DataFrame.hpp` | Typed columns, basic CSV loading, and one-hot encoding |
| `LinearRegression`, `Scaler` | `cpp_ml/models/LinearRegression.hpp` | Gradient-descent regression and feature standardization |
| `LogisticRegression` | `cpp_ml/models/LogisticRegression.hpp` | Binary probabilities and classification metrics |
| `DecisionTreeClassifier` | `cpp_ml/models/DecisionTreeClassifier.hpp` | Binary classification using entropy-based splits |
| `MLPClassifier` | `cpp_ml/models/MLPClassifier.hpp` | Fully connected binary classifier with ReLU hidden layers |

## Repository layout

```text
include/cpp_ml/
  core/             Matrix kernels, random utilities, and profiling
  data/             DataFrame and CSV reader
  models/           Regression, decision tree, and MLP implementations
examples/           Runnable model examples
benchmarks/         MLP driver, Python tools, and historical results
tests/              Kernel and model correctness checks
data/               Existing sample datasets and provenance notes
docs/               Build, architecture, and performance notes
```

## Documentation and contribution

- [Documentation index](docs/README.md)
- [Architecture and current limitations](docs/architecture.md)
- [Build options and verification](docs/building.md)
- [Benchmarks and reproduction](docs/benchmarks.md)
- [Examples](examples/README.md) and [dataset notes](data/README.md)
- [Contributing](CONTRIBUTING.md)

Contributions to correctness, examples, documentation, portability, and performance
are welcome. Start with the contribution guide and open an issue for a substantial
API or design change.

## Project status

The API is evolving. CSV support is basic, and model validation, preprocessing,
and tree memory ownership need further work. See the architecture notes for known
limitations. Tests cover matrix kernels, MLP reference agreement, and basic
regression learning; they do not establish correctness of every component.

## License

The source code is distributed under the [MIT license](LICENSE).
See [dataset notes](data/README.md) for the separate provenance status of CSV files.

## AI assistance

AI-assisted contributions are welcome. Contributors are responsible for
understanding, reviewing, and validating their changes.
