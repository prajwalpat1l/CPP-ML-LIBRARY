# Architecture

CPP-ML-LIBRARY is header-only. CMake exposes the `cpp_ml::cpp_ml` interface target,
which supplies the include directory, C++20 requirement, numerical compiler options,
and optional OpenMP linkage. No compiled library is required.

## Code map

- `include/cpp_ml/core/Matrix.hpp` owns row-major `double` storage and implements
  matrix operations plus reusable training kernels. Public indexing checks bounds;
  internal kernels also use raw contiguous buffers.
- `core/Random.hpp` supplies reproducible random helpers. `core/Profiler.hpp`
  enables operation-level timing when `ML_PROFILE` is defined.
- `data/DataFrame.hpp` contains typed columns, the DataFrame, basic CSV parsing,
  and one-hot encoding. `to_matrix()` bridges tabular data into models.
- `models/` contains independent model headers. `Scaler` currently shares the
  linear-regression header. Most model APIs use `fit()` and `predict()`; argument
  order and probability/label conventions are not yet uniform.
- The MLP uses reusable activation and gradient buffers. OpenMP kernels split
  work across threads. The default gradient split preserves summation order;
  `GradSplit::rows` regroups sums and can change floating-point results.

Classes currently use the global namespace. Include paths moved from bare header
names to `cpp_ml/core/...`, `cpp_ml/data/...`, and `cpp_ml/models/...` in the
repository cleanup. Consumers of the old layout must update their includes.

## Typical workflow

Load or construct data, select numeric features, split training and evaluation
rows, fit preprocessing on training rows, transform both splits, train a model,
and evaluate its predictions. The synthetic regression example avoids data-loading
and preprocessing so that a first build stays simple.

## Current limitations

- CSV parsing is a simple comma splitter; quoted fields, escaped quotes, missing
  values, and scientific-notation type inference need work.
- One-hot encoding can overwrite existing columns with matching output names.
- Repeated `Scaler::fit()` calls retain previous accumulations. Use a fresh scaler
  for each fit until this is fixed. Variance calculation also needs stability work.
- The decision tree owns raw node pointers without cleanup; repeated fitting and
  destruction need proper ownership handling.
- Input validation and unfitted-model behavior are incomplete across models.
- The MLP and decision tree currently support binary classification. Model
  serialization and a stable cross-model API are not provided.
- The test suite emphasizes numerical kernels and MLP reference comparisons;
  DataFrame, scaler, and decision-tree behavior need dedicated coverage.

These limitations are documented to guide future fixes; the structural cleanup
preserves the existing model algorithms.
