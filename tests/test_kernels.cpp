// Checks the Matrix kernels and MLPClassifier training against plain loops,
// on shapes the benchmark never uses: row counts that are not multiples of 4,
// very narrow and very wide matrices, 1 to 8 threads, and shapes on both sides
// of each kernel's switch from one thread to all of them. Every check except
// matmul_tn_reduce must be bit-identical, since the kernels do the same
// additions in the same order. Exits non-zero on any failure.
//
// With ML_FMA on the compiler may fuse the kernels and the references
// differently, so exact checks become tolerance checks (ML_TEST_TOLERANT).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "cpp_ml/core/Matrix.hpp"
#include "cpp_ml/models/MLPClassifier.hpp"
#include "cpp_ml/models/LinearRegression.hpp"
#include "cpp_ml/models/LogisticRegression.hpp"
#include "cpp_ml/core/Random.hpp"

namespace {

int checks = 0, failures = 0;

void fail(const std::string& what) {
    failures++;
    if (failures <= 30) {
        std::printf("FAIL  %s\n", what.c_str());
    }
}

std::string shape(const char* name, size_t a, size_t b, size_t c, int threads) {
    return std::string(name) + " " + std::to_string(a) + "x" + std::to_string(b) + "x" + std::to_string(c) + " t" + std::to_string(threads);
}

void set_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

// Values in [-1, 1), with about one in eight exactly zero so relu masks and
// signed zeros get exercised.
Matrix random_matrix(size_t rows, size_t cols, std::mt19937& gen) {
    Matrix M(rows, cols);
    for (size_t i = 0; i < M.size(); i++) {
        M.raw()[i] = (gen() % 8 == 0) ? 0.0 : rng::uniform(gen, -1.0, 1.0);
    }
    return M;
}

// Filled with NaN, so a kernel that forgets to initialize part of its output fails.
Matrix garbage(size_t rows, size_t cols) {
    Matrix M(rows, cols);
    std::fill(M.raw(), M.raw() + M.size(), std::numeric_limits<double>::quiet_NaN());
    return M;
}

// Bitwise equality (so 0.0 vs -0.0 and NaN count as differences), or close
// within `tol` relative to the reference's magnitude under ML_TEST_TOLERANT.
void expect_same(const Matrix& got, const Matrix& want, const std::string& what, double tol = 1e-12) {
    checks++;
    if (got.rows() != want.rows() || got.cols() != want.cols()) {
        fail(what + ": wrong shape");
        return;
    }
#ifdef ML_TEST_TOLERANT
    for (size_t i = 0; i < got.size(); i++) {
        double g = got.raw()[i], w = want.raw()[i];
        if (!(std::abs(g - w) <= tol * (1.0 + std::abs(w)))) {
            fail(what + ": element " + std::to_string(i) + " differs");
            return;
        }
    }
#else
    (void)tol;
    if (std::memcmp(got.raw(), want.raw(), got.size() * sizeof(double)) != 0) {
        size_t i = 0;
        while (std::memcmp(got.raw() + i, want.raw() + i, sizeof(double)) == 0) {
            i++;
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, ": element %zu is %.17g, expected %.17g", i, got.raw()[i], want.raw()[i]);
        fail(what + buf);
    }
#endif
}

// Reference kernels: one output at a time, terms in index order.

Matrix ref_matmul(const Matrix& A, const Matrix& B, const Matrix* bias = nullptr) {
    Matrix C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < B.cols(); j++) {
            double s = bias ? (*bias)(0, j) : 0.0;
            for (size_t k = 0; k < A.cols(); k++) {
                s += A(i, k) * B(k, j);
            }
            C(i, j) = s;
        }
    }
    return C;
}

Matrix ref_matmul_tn(const Matrix& A, const Matrix& B) {
    Matrix C(A.cols(), B.cols());
    for (size_t j = 0; j < A.cols(); j++) {
        for (size_t k = 0; k < B.cols(); k++) {
            double s = 0.0;
            for (size_t i = 0; i < A.rows(); i++) {
                s += A(i, j) * B(i, k);
            }
            C(j, k) = s;
        }
    }
    return C;
}

// Sum of |terms| per output, the scale of rounding error for a reordered sum.
Matrix ref_matmul_tn_abs(const Matrix& A, const Matrix& B) {
    Matrix C(A.cols(), B.cols());
    for (size_t j = 0; j < A.cols(); j++) {
        for (size_t k = 0; k < B.cols(); k++) {
            double s = 0.0;
            for (size_t i = 0; i < A.rows(); i++) {
                s += std::abs(A(i, j) * B(i, k));
            }
            C(j, k) = s;
        }
    }
    return C;
}

Matrix ref_column_sums(const Matrix& A) {
    Matrix C(1, A.cols());
    for (size_t j = 0; j < A.cols(); j++) {
        double s = 0.0;
        for (size_t i = 0; i < A.rows(); i++) {
            s += A(i, j);
        }
        C(0, j) = s;
    }
    return C;
}

// Kernel tests

void test_matmuls(size_t m, size_t n, size_t q, int threads, std::mt19937& gen) {
    set_threads(threads);
    Matrix A = random_matrix(m, n, gen);
    Matrix B = random_matrix(n, q, gen);
    Matrix bias = random_matrix(1, q, gen);

    Matrix out = garbage(m, q);
    matmul(A, B, out);
    expect_same(out, ref_matmul(A, B), shape("matmul", m, n, q, threads));

    out = garbage(m, q);
    matmul_bias(A, B, bias, out);
    expect_same(out, ref_matmul(A, B, &bias), shape("matmul_bias", m, n, q, threads));

    // Aᵀ·B with A: m×n and B2: m×q.
    Matrix B2 = random_matrix(m, q, gen);
    Matrix want = ref_matmul_tn(A, B2);
    Matrix tn = garbage(n, q);
    matmul_tn(A, B2, tn);
    expect_same(tn, want, shape("matmul_tn", m, n, q, threads));

    // The reduce variant regroups the sum unless there's one thread: exact at
    // t1, otherwise within rounding of the reference and the same every run.
    ThreadPartials partials;
    Matrix red = garbage(n, q);
    matmul_tn_reduce(A, B2, red, partials);
    if (threads == 1) {
        expect_same(red, want, shape("matmul_tn_reduce", m, n, q, threads));
    } else {
        checks++;
        Matrix scale = ref_matmul_tn_abs(A, B2);
        double eps = std::numeric_limits<double>::epsilon();
        for (size_t e = 0; e < red.size(); e++) {
            double bound = (m + threads) * eps * scale.raw()[e];
            if (!(std::abs(red.raw()[e] - want.raw()[e]) <= bound)) {
                fail(shape("matmul_tn_reduce", m, n, q, threads) + ": element " + std::to_string(e) + " outside rounding bound");
                break;
            }
        }
        Matrix again = garbage(n, q);
        matmul_tn_reduce(A, B2, again, partials);
        checks++;
        if (std::memcmp(again.raw(), red.raw(), red.size() * sizeof(double)) != 0) {
            fail(shape("matmul_tn_reduce", m, n, q, threads) + ": not deterministic");
        }
    }
}

void test_column_sums_and_transpose(size_t m, size_t n, int threads, std::mt19937& gen) {
    set_threads(threads);
    Matrix A = random_matrix(m, n, gen);

    Matrix sums = garbage(1, n);
    column_sums(A, sums);
    expect_same(sums, ref_column_sums(A), shape("column_sums", m, n, 1, threads));

    Matrix t = garbage(n, m);
    transpose(A, t);
    expect_same(t, A.transpose(), shape("transpose", m, n, 1, threads));
}

void test_argument_checks() {
    Matrix A(5, 3), B(3, 2), out(5, 2), wrong(2, 5), square(3, 3);
    auto expect_throw = [](const char* what, auto&& f) {
        checks++;
        try {
            f();
            fail(std::string(what) + ": did not throw");
        } catch (const std::invalid_argument&) {
        }
    };
    expect_throw("matmul inner mismatch", [&] { matmul(A, A, out); });
    expect_throw("matmul wrong out shape", [&] { matmul(A, B, wrong); });
    expect_throw("matmul out aliases input", [&] { matmul(square, square, square); });
    expect_throw("matmul_tn row mismatch", [&] { Matrix o(3, 2); matmul_tn(A, B, o); });
    expect_throw("matmul_bias bias shape", [&] { Matrix b(2, 2); matmul_bias(A, B, b, out); });
    expect_throw("column_sums wrong out", [&] { Matrix o(1, 2); column_sums(A, o); });
}

void test_mlp(size_t m, size_t d, const std::vector<size_t>& neurons, std::mt19937& gen);

// Thread counts

void expect_threads(int got, int want, const std::string& what) {
#ifndef _OPENMP
    want = 1; // Serial builds always use one thread, regardless of the request.
#endif
    checks++;
    if (got != want) {
        fail(what + ": chose " + std::to_string(got) + " threads, expected " + std::to_string(want));
    }
}

// One thread or all of them, never a count in between.
void test_choose_threads() {
    set_threads(8);
    expect_threads(choose_threads(0, 100), 1, "choose_threads no work");
    expect_threads(choose_threads(199, 100), 1, "choose_threads under 2 shares");
    expect_threads(choose_threads(200, 100), 8, "choose_threads 2 shares");
    expect_threads(choose_threads(100000, 100), 8, "choose_threads many shares");
    expect_threads(choose_threads(100000, 100, 5), 8, "choose_threads 5 parts");
    expect_threads(choose_threads(100000, 100, 1), 1, "choose_threads 1 part");
    expect_threads(choose_threads(100000, 100, 0), 1, "choose_threads no parts");
    set_threads(3);
    expect_threads(choose_threads(100000, 100), 3, "choose_threads capped by OMP max");
}

// The smallest row count m ≡ 3 (mod 4), so leftover rows run too, at which
// choice(m) is the maximum thread count; choice(m - 4) must still be 1.
template <class Choice>
size_t threshold_rows(Choice choice, const std::string& what) {
    int max_threads = choose_threads(SIZE_MAX, 1);
    for (size_t m = 3; m < 100000000; m += 4) {
        if (choice(m) != 1) {
            expect_threads(choice(m), max_threads, what + " rows " + std::to_string(m));
            if (m > 3) {
                expect_threads(choice(m - 4), 1, what + " rows " + std::to_string(m - 4));
            }
            return m;
        }
    }
    fail(what + ": never threaded");
    return 3;
}

// Each kernel on both sides of the shape where it switches from one thread
// to the maximum, found from the tuned minimums so the shapes follow them.
void test_thresholds(std::mt19937& gen) {
    for (int threads : {2, 8}) {
        set_threads(threads);
        std::string tag = " (threshold, t" + std::to_string(threads) + ")";

        size_t m = threshold_rows([](size_t r) { return thread_count::matmul(r, 33, 17); }, "matmul" + tag);
        test_matmuls(m - 4, 33, 17, threads, gen);
        test_matmuls(m, 33, 17, threads, gen);

        m = threshold_rows([](size_t r) { return thread_count::matmul_tn_rows(r, 33, 17); }, "matmul_tn_reduce" + tag);
        test_matmuls(m - 4, 33, 17, threads, gen);
        test_matmuls(m, 33, 17, threads, gen);

        // matmul_tn split outputs, limited by gradient rows: 2 shares (threaded) and just under.
        m = 4001;
        size_t p = 2 * min_grad_rows_per_thread;
        expect_threads(thread_count::matmul_tn_outputs(m, p, 33), threads, "matmul_tn p " + std::to_string(p) + tag);
        test_matmuls(m, p, 33, threads, gen);
        if (p > 1) {
            expect_threads(thread_count::matmul_tn_outputs(m, p - 1, 33), 1, "matmul_tn p " + std::to_string(p - 1) + tag);
            test_matmuls(m, p - 1, 33, threads, gen);
        }

        m = threshold_rows([](size_t r) { return thread_count::column_sums(r, 33); }, "column_sums" + tag);
        test_column_sums_and_transpose(m - 4, 33, threads, gen);
        test_column_sums_and_transpose(m, 33, threads, gen);

        // A model whose 33-wide relu layer switches to threads (trained at 1, 3 and 8).
        if (threads == 8) {
            m = threshold_rows([](size_t r) { return thread_count::elementwise(r * 33); }, "relu" + tag);
            test_mlp(m - 4, 8, {33, 17, 1}, gen);
            test_mlp(m, 8, {33, 17, 1}, gen);
        }
    }
}

// Reference MLP: the same algorithm as MLPClassifier, written plainly.

class ReferenceMLP {
private:
    struct Layer {
        Matrix W, b, Z, A;
    };
    std::vector<Layer> layers;

    void forward(const Matrix& X) {
        const Matrix* in = &X;
        for (size_t l = 0; l < layers.size(); l++) {
            Layer& L = layers[l];
            L.Z = ref_matmul(*in, L.W, &L.b);
            L.A = Matrix(L.Z.rows(), L.Z.cols());
            for (size_t e = 0; e < L.Z.size(); e++) {
                double z = L.Z.raw()[e];
                L.A.raw()[e] = (l + 1 == layers.size()) ? 1.0 / (1.0 + std::exp(-z)) : std::max(z, 0.0);
            }
            in = &L.A;
        }
    }

public:
    ReferenceMLP(size_t input_size, const std::vector<size_t>& neurons, std::uint32_t seed) {
        std::mt19937 gen(seed);
        layers.resize(neurons.size());
        for (size_t l = 0; l < neurons.size(); l++) {
            size_t n_in = (l == 0) ? input_size : neurons[l - 1];
            layers[l].W = Matrix(n_in, neurons[l]);
            layers[l].b = Matrix(1, neurons[l]);
        }
        // Same draw order as MLPClassifier: each layer's biases, then W by rows.
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

    void fit(const Matrix& X, const Matrix& y, size_t epochs, double lr) {
        double inv_m = 1.0 / static_cast<double>(y.rows());
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            forward(X);
            Matrix dZ = layers.back().A - y;
            for (int l = static_cast<int>(layers.size()) - 1; l >= 0; l--) {
                Layer& L = layers[l];
                const Matrix& a_prev = (l == 0) ? X : layers[l - 1].A;
                Matrix dW = ref_matmul_tn(a_prev, dZ);
                Matrix db = ref_column_sums(dZ);
                Matrix dA;
                if (l > 0) {
                    dA = ref_matmul(dZ, L.W.transpose());
                    const Matrix& Z = layers[l - 1].Z;
                    for (size_t e = 0; e < dA.size(); e++) {
                        dA.raw()[e] *= (Z.raw()[e] > 0) ? 1.0 : 0.0;
                    }
                }
                for (size_t e = 0; e < L.W.size(); e++) {
                    L.W.raw()[e] = L.W.raw()[e] - (dW.raw()[e] * inv_m) * lr;
                }
                for (size_t e = 0; e < L.b.size(); e++) {
                    L.b.raw()[e] = L.b.raw()[e] - (db.raw()[e] * inv_m) * lr;
                }
                dZ = dA;
            }
        }
    }

    Matrix predict(const Matrix& X) {
        forward(X);
        return layers.back().A;
    }
};

std::string mlp_name(size_t m, size_t d, const std::vector<size_t>& neurons, int threads, const char* split) {
    std::string s = "mlp " + std::to_string(m) + " rows, " + std::to_string(d);
    for (size_t n : neurons) {
        s += "-" + std::to_string(n);
    }
    return s + " t" + std::to_string(threads) + " " + split;
}

// Trains on m rows, then predicts on a different row count so predict()
// resizes the buffers. Learning rate 0.5 so a few epochs move every weight.
void test_mlp(size_t m, size_t d, const std::vector<size_t>& neurons, std::mt19937& gen) {
    size_t epochs = 3;
    double lr = 0.5;
    Matrix X = random_matrix(m, d, gen);
    Matrix y(m, 1);
    for (size_t i = 0; i < m; i++) {
        y(i, 0) = (gen() % 2) ? 1.0 : 0.0;
    }
    size_t m_test = m + 3;
    Matrix X_test = random_matrix(m_test, d, gen);

    ReferenceMLP ref(d, neurons, 7);
    ref.fit(X, y, epochs, lr);
    Matrix want_train = ref.predict(X);
    Matrix want_test = ref.predict(X_test);

    Matrix rows_t1;
    for (int threads : {1, 3, 8}) {
        set_threads(threads);
        MLPClassifier model(d, neurons, 7);
        model.fit(X, y, epochs, lr);
        expect_same(model.predict(X), want_train, mlp_name(m, d, neurons, threads, "outputs") + " (train rows)");
        expect_same(model.predict(X_test), want_test, mlp_name(m, d, neurons, threads, "outputs") + " (test rows)");

        // GradSplit::rows: exact at one thread, close and repeatable otherwise.
        Matrix first;
        for (int run = 0; run < 2; run++) {
            MLPClassifier r(d, neurons, 7);
            r.set_grad_split(GradSplit::rows);
            r.fit(X, y, epochs, lr);
            Matrix p = r.predict(X_test);
            if (run == 0) {
                first = p;
            } else {
                checks++;
                if (std::memcmp(first.raw(), p.raw(), p.size() * sizeof(double)) != 0) {
                    fail(mlp_name(m, d, neurons, threads, "rows") + ": not deterministic");
                }
            }
        }
        if (threads == 1) {
            expect_same(first, want_test, mlp_name(m, d, neurons, threads, "rows"));
        } else {
            checks++;
            for (size_t e = 0; e < first.size(); e++) {
                if (!(std::abs(first.raw()[e] - want_test.raw()[e]) <= 1e-9)) {
                    fail(mlp_name(m, d, neurons, threads, "rows") + ": not close to reference");
                    break;
                }
            }
        }
    }
}

// Linear and logistic regression share Matrix, so check they still learn.

void test_regressions(std::mt19937& gen) {
    set_threads(8);
    size_t m = 1001;
    Matrix X = random_matrix(m, 2, gen);
    Matrix y(m, 1), labels(m, 1);
    for (size_t i = 0; i < m; i++) {
        y(i, 0) = 1.5 + 2.0 * X(i, 0) - 3.0 * X(i, 1);
        labels(i, 0) = (X(i, 0) - X(i, 1) > 0.1) ? 1.0 : 0.0;
    }

    LinearRegression lin;
    lin.fit(X, y, 0.5, 2000);
    double mse = LinearRegression::MSE_Error(y, lin.predict(X));
    checks++;
    if (!(mse < 1e-12)) {
        fail("LinearRegression: MSE " + std::to_string(mse) + " on noise-free data");
    }

    LogisticRegression logit;
    logit.fit(X, labels, 1.0, 2000);
    Matrix p = logit.predict(X);
    double acc = LogisticRegression::accuracy(labels, p);
    double loss = LogisticRegression::cross_entropy_loss(labels, p);
    checks++;
    if (!(acc > 97.0 && loss < 0.2)) {
        fail("LogisticRegression: accuracy " + std::to_string(acc) + "%, loss " + std::to_string(loss));
    }
}

}

int main() {
    std::mt19937 gen(12345);

    // 1-9 cover every remainder mod 4 and fewer rows than threads; with 4001
    // rows the elementwise ops and the wider multiplies get several threads.
    std::vector<size_t> rows = {1, 2, 3, 4, 5, 7, 8, 9, 4001};
    std::vector<size_t> widths = {1, 3, 8, 33};
    for (int threads : {1, 2, 3, 4, 8}) {
        for (size_t m : rows) {
            for (size_t n : widths) {
                for (size_t q : widths) {
                    test_matmuls(m, n, q, threads, gen);
                }
            }
            // Very narrow next to very wide.
            test_matmuls(m, 1, 600, threads, gen);
            test_matmuls(m, 600, 1, threads, gen);
            test_matmuls(m, 300, 300, threads, gen);
            for (size_t n : {1, 3, 33, 255, 256, 257, 600}) {
                test_column_sums_and_transpose(m, n, threads, gen);
            }
        }
        // Over 256 columns per thread, so column_sums takes two passes over the rows.
        test_column_sums_and_transpose(4001, 2100, threads, gen);
    }
    test_argument_checks();

    std::vector<std::vector<size_t>> nets = {{1}, {33, 17, 1}, {300, 1}, {2, 1}, {5, 600, 3, 1}};
    for (size_t m : {1, 3, 5, 7, 4001}) {
        for (size_t d : {1, 8, 300}) {
            for (const auto& neurons : nets) {
                test_mlp(m, d, neurons, gen);
            }
        }
    }

    test_choose_threads();
#ifdef _OPENMP
    test_thresholds(gen);
#endif

    test_regressions(gen);

    std::printf("%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
