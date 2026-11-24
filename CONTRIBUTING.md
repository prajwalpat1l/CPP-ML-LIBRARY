# Contributing

Thanks for helping improve CPP-ML-LIBRARY. Useful contributions include reproducible
bug fixes, numerical checks, clearer examples, documentation, and build portability.

## Set up

1. Fork the repository and clone your fork.
2. Create a branch for one coherent improvement.
3. Follow [the build instructions](docs/building.md), then run the tests.
4. Read [the architecture notes](docs/architecture.md) before changing model behavior.

## Make a change

- Keep public headers under `include/cpp_ml/` and use the `cpp_ml/...` include prefix.
- Use four-space indentation and follow the surrounding C++ style.
- Keep generated builds, local benchmark results, and downloaded tools out of Git.
- For behavior changes, add a focused test that demonstrates the intended result.
- For performance changes, report compiler, hardware, flags, thread count, and
  repeated measurements. Verify predictions as well as speed.
- Update examples and documentation when changing public APIs or paths.
- Discuss new algorithms or broad redesigns in an issue before implementing them.

## Submit a pull request

Explain the problem, what changes for users, and how you verified it. Include
exact commands and relevant results. Link an issue if one exists. Keep unrelated
refactors separate so the change is easy to review.

AI-assisted contributions are welcome. Understand the submitted code, disclose
substantial assistance in the PR, verify the results, and address review feedback.
Preserve existing copyright notices and credit sources where applicable.

## Report a bug

Include a minimal reproducer, expected and actual results, operating system,
compiler version, CMake options, and whether OpenMP is enabled. Use a small
synthetic dataset where possible.
