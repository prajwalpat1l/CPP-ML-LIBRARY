#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include "cpp_ml/core/Matrix.hpp"
#include "cpp_ml/core/Random.hpp"
#include "cpp_ml/core/Profiler.hpp"
#include "cpp_ml/models/LogisticRegression.hpp"

// How backward threads the weight gradient dW = Aᵀ·dZ.
// outputs (default): threads split the rows of dW. Bit-identical to one thread.
// rows: threads split the sum over samples (see matmul_tn_reduce). Close but
// not bit-identical; about 1.4x faster on the small benchmark net at 8 threads,
// whose first-layer dW has only 8 rows to split.
enum class GradSplit {
    outputs,
    rows
};

class MLPClassifier {
private:
    // Z, A and dZ have one row per sample and are resized only when the
    // number of rows changes.
    struct Layer {
        Matrix W, b;   // n_in × n_out, 1 × n_out
        Matrix Z, A;   // m × n_out
        Matrix dZ;     // m × n_out
        Matrix dW, db; // unscaled gradients
        Matrix Wt;     // n_out × n_in, for dA = dZ·Wᵀ
    };

    std::vector<Layer> layers;
    std::mt19937 gen;
    GradSplit grad_split = GradSplit::outputs;
    ThreadPartials partials; // for GradSplit::rows

    void relu(const Matrix& Z, Matrix& A) {
        ML_PROFILE_SCOPE("MLP::relu");
        const double* z = Z.raw();
        double* a = A.raw();
        int threads = thread_count::elementwise(Z.size());
        ML_PROFILE_THREADS(threads);
        #pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
        for (size_t i = 0; i < Z.size(); i++) {
            a[i] = std::max(z[i], 0.0);
        }
    }

    void sigmoid(const Matrix& Z, Matrix& A) {
        ML_PROFILE_SCOPE("MLP::sigmoid");
        const double* z = Z.raw();
        double* a = A.raw();
        int threads = thread_count::elementwise(Z.size());
        ML_PROFILE_THREADS(threads);
        #pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
        for (size_t i = 0; i < Z.size(); i++) {
            a[i] = 1.0 / (1.0 + std::exp(-z[i]));
        }
    }

    // dZ *= relu'(Z), in place. Z is overwritten with the 0/1 mask, which is
    // fine since nothing reads it before the next forward pass. A multiply,
    // not a select, so negative gradients become -0.0 as before.
    // Two loops on purpose: fused, GCC branches on the sign of z, which
    // mispredicts half the time (~10x slower). Both use the same static
    // schedule, so each thread gets the same elements and the first loop
    // needs no barrier.
    void relu_backward(Matrix& dZ, Matrix& Z) {
        ML_PROFILE_SCOPE("MLP::relu_backward");
        double* dz = dZ.raw();
        double* z = Z.raw();
        size_t n = dZ.size();
        int threads = thread_count::elementwise(n);
        ML_PROFILE_THREADS(threads);
        #pragma omp parallel num_threads(threads) if (threads > 1)
        {
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; i++) {
                z[i] = ((z[i] > 0) ? 1.0: 0.0);
            }
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; i++) {
                dz[i] *= z[i];
            }
        }
    }

    // P = P - (grad * inv_m) * lr, in the same order as the old
    // P - (grad * (1/m)) * lr.
    void sgd_step(Matrix& P, const Matrix& grad, const double inv_m, const double learning_rate) {
        ML_PROFILE_SCOPE("MLP::sgd_step");
        double* p = P.raw();
        const double* g = grad.raw();
        int threads = thread_count::elementwise(P.size());
        ML_PROFILE_THREADS(threads);
        #pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
        for (size_t i = 0; i < P.size(); i++) {
            p[i] = p[i] - (g[i] * inv_m) * learning_rate;
        }
    }

    void resize_buffers(const size_t rows) {
        for (Layer& L : layers) {
            if (L.Z.rows() != rows) {
                L.Z.resize(rows, L.W.cols());
                L.A.resize(rows, L.W.cols());
                L.dZ.resize(rows, L.W.cols());
            }
        }
    }

    void forward(const Matrix& X) {
        ML_PROFILE_SCOPE("MLP::forward");
        resize_buffers(X.rows());
        const Matrix* in = &X;
        for (size_t i = 0; i < layers.size(); i++) {
            Layer& L = layers[i];
            matmul_bias(*in, L.W, L.b, L.Z);
            if (i != layers.size() - 1) {
                relu(L.Z, L.A);
            } else {
                sigmoid(L.Z, L.A);
            }
            in = &L.A;
        }
    }

    void backward(const Matrix& X, const Matrix& y, const double learning_rate) {
        ML_PROFILE_SCOPE("MLP::backward");
        double m = static_cast<double>(y.rows());
        double inv_m = 1.0/m;

        Layer& out = layers.back();
        if (y.rows() != out.A.rows() || y.cols() != out.A.cols()) {
            throw std::invalid_argument("Target matrix has the wrong shape.");
        }
        for (size_t i = 0; i < y.size(); i++) {
            out.dZ.raw()[i] = out.A.raw()[i] - y.raw()[i];
        }

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer& L = layers[i];
            const Matrix& A_prev = (i == 0) ? X : layers[i - 1].A;
            if (grad_split == GradSplit::rows) {
                matmul_tn_reduce(A_prev, L.dZ, L.dW, partials);
            } else {
                matmul_tn(A_prev, L.dZ, L.dW);
            }
            column_sums(L.dZ, L.db);

            // dA uses W from before this step's update.
            if (i > 0) {
                Layer& prev = layers[i - 1];
                transpose(L.W, L.Wt);
                matmul(L.dZ, L.Wt, prev.dZ);
                relu_backward(prev.dZ, prev.Z);
            }

            sgd_step(L.W, L.dW, inv_m, learning_rate);
            sgd_step(L.b, L.db, inv_m, learning_rate);
        }
    }

public:
    MLPClassifier(size_t input_size, std::vector<size_t> neurons = {1}, std::uint32_t seed = 42) : gen(seed) {
        if (neurons.empty()) {
            throw std::invalid_argument("The neural network cannot be empty.");
        }
        if (neurons.back() != 1) {
            throw std::invalid_argument("The output layer should only have one neuron.");
        }

        layers.resize(neurons.size());
        for (size_t i = 0; i < layers.size(); i++) {
            size_t n_in = (i == 0) ? input_size : neurons[i - 1];
            layers[i].W = Matrix(n_in, neurons[i]);
            layers[i].b = Matrix(1, neurons[i]);
            layers[i].dW = Matrix(n_in, neurons[i]);
            layers[i].db = Matrix(1, neurons[i]);
            layers[i].Wt = Matrix(neurons[i], n_in);
        }

        // Biases first, then W row by row: the draw order of the old combined
        // matrix, whose first row held the biases.
        for (Layer& L : layers) {
            for (size_t k = 0; k < L.b.cols(); k++) {
                L.b(0, k) = rng::uniform(gen, -0.1, 0.1);
            }
            for (size_t j = 0; j < L.W.rows(); j++) {
                for (size_t k = 0; k < L.W.cols(); k++) {
                    L.W(j, k) = rng::uniform(gen, -0.1, 0.1);
                }
            }
        }
    }

    void set_grad_split(GradSplit split) {
        grad_split = split;
    }

    void fit(const Matrix& X, const Matrix& y, const size_t epochs, const double learning_rate = 0.01) {
        ML_PROFILE_SCOPE("MLP::fit");
        size_t iterations = 0;
        while (iterations++ != epochs) {
            forward(X);
            backward(X, y, learning_rate);
        }
    }

    Matrix predict(const Matrix& X) {
        ML_PROFILE_SCOPE("MLP::predict");
        forward(X);
        return layers.back().A;
    }
};
