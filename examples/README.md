# Examples

Build with `ML_BUILD_EXAMPLES=ON` (the default). Run from the repository root.

## Synthetic linear regression

```sh
./build/linear_regression
```

Learns `y = 1 + 2x` using five constructed rows and predicts `7` for `x = 3`.
No CSV or Python dependencies are needed. The executable returns a nonzero status
if its prediction falls outside the expected tolerance.

## Diabetes MLP

```sh
./build/diabetes_mlp
```

The dataset-based example reads `data/Diabetes_Prediction.csv`, shuffles
rows, encodes gender, standardizes features, and trains a binary MLP. It uses
8,000 training rows, 2,000 test rows, and 10,000 epochs, so it can take considerably
longer than the synthetic example. It prints test accuracy as a percentage.
This is an educational example, not a clinical prediction tool.

See [dataset notes](../data/README.md) and [build instructions](../docs/building.md).
