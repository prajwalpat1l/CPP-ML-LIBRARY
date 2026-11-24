#pragma once

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cpp_ml/core/Profiler.hpp"

class Matrix {
private:
    std::vector<double> data;
    size_t n_rows;
    size_t n_cols;

public:
    Matrix() : n_rows(0), n_cols(0) {}

    Matrix(size_t rows, size_t cols) : data(rows*cols), n_rows(rows), n_cols(cols) {}

    size_t rows() const {
        return n_rows;
    }

    size_t cols() const {
        return n_cols;
    }

    // Unchecked access for the library's own loops. operator() stays checked.
    double* raw() {
        return data.data();
    }

    const double* raw() const {
        return data.data();
    }

    size_t size() const {
        return data.size();
    }

    double operator()(size_t i, size_t j) const {
        if (i >= n_rows || j >= n_cols) {
            throw std::invalid_argument("Matrix indexing out of bounds.");
        }
        return data[i*n_cols + j];
    }

    double& operator()(size_t i, size_t j) {
        if (i >= n_rows || j >= n_cols) {
            throw std::invalid_argument("Matrix indexing out of bounds.");
        }

        return data[i * n_cols + j];
    }

    Matrix operator+(const Matrix& other) const { // Revise the operator overloading here.
        ML_PROFILE_SCOPE("Matrix::add");
        if (n_rows != other.n_rows || n_cols != other.n_cols) {
            throw std::invalid_argument("Matrix sizes do not match.");
        }

        Matrix result(n_rows, n_cols);
        for (size_t i = 0; i < n_rows*n_cols; i++) {
            result.data[i] = this -> data[i] + other.data[i];
        }

        return result;
    }

    Matrix operator-(const Matrix& other) const { // Revise the operator overloading here.
        ML_PROFILE_SCOPE("Matrix::sub");
        if (n_rows != other.n_rows || n_cols != other.n_cols) {
            throw std::invalid_argument("Matrix sizes do not match.");
        }

        Matrix result(n_rows, n_cols);
        for (size_t i = 0; i < n_rows*n_cols; i++) {
            result.data[i] = this -> data[i] - other.data[i];
        }

        return result;
    }

    Matrix operator*(const double& other) const {
        ML_PROFILE_SCOPE("Matrix::scale");
        Matrix result(n_rows, n_cols);
        for (size_t i = 0; i < n_rows*n_cols; i++) {
            result.data[i] = other*this -> data[i];
        }

        return result;
    }

    Matrix transpose() const {
        ML_PROFILE_SCOPE("Matrix::transpose");
        Matrix result(n_cols, n_rows);
        for (size_t i = 0; i < n_rows; i++) {
            for (size_t j = 0; j < n_cols; j++) {
                result.data[j*n_rows + i] = data[i*n_cols + j];
            }
        }

        return result;
    }

    Matrix operator*(const Matrix& other) const {
        ML_PROFILE_SCOPE("Matrix::matmul");
        if (this -> n_cols != other.n_rows) {
            throw std::invalid_argument("Matrix dimensions are not valid.");
        }
        ML_PROFILE_FLOPS(2 * n_rows * n_cols * other.n_cols);

        Matrix result(n_rows, other.n_cols);
        for (size_t i = 0; i < n_rows; i++) {
            for (size_t j = 0; j < other.n_cols; j++) {
                double sum = 0;
                for (size_t k = 0; k < this -> n_cols; k++) {
                    sum += (this -> data[i*n_cols + k])*(other.data[k*other.n_cols + j]);
                }
                result.data[i*other.n_cols + j] = sum;
            }
        }

        return result;
    }

    Matrix operator%(const Matrix& other) const {
        ML_PROFILE_SCOPE("Matrix::hadamard");
        if (this -> n_rows != other.n_rows || this -> n_cols != other.n_cols) {
            throw std::invalid_argument("Matrix dimensions do not match for Hadamard product.");
        }

        Matrix result(n_rows, n_cols);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] * other.data[i];
        }

        return result;
    }

    Matrix add_bias_column() const {
        ML_PROFILE_SCOPE("Matrix::add_bias_column");
        Matrix result(n_rows, n_cols + 1);

        for (size_t i = 0; i < n_rows; i++) {
            double* out = result.data.data() + i * (n_cols + 1);
            const double* in = data.data() + i * n_cols;

            out[0] = 1.0;

            for (size_t j = 0; j < n_cols; j++) {
                out[j + 1] = in[j];
            }
        }

        return result;
    }

    Matrix remove_bias_column() const {
        ML_PROFILE_SCOPE("Matrix::remove_bias_column");
        Matrix result(n_rows, n_cols - 1);

        for (size_t i = 0; i < n_rows; i++) {
            double* out = result.data.data() + i * (n_cols - 1);
            const double* in = data.data() + i * n_cols;
            for (size_t j = 1; j < n_cols; j++) {
                out[j - 1] = in[j];
            }
        }

        return result;
    }

    Matrix slice(size_t start, size_t end) const {
        ML_PROFILE_SCOPE("Matrix::slice");
        if (start >= end || end > n_rows) {
            throw std::invalid_argument("Input parameters are out of range.");
        }

        Matrix result(end - start, n_cols);
        const double* in = data.data() + start * n_cols;
        for (size_t i = 0; i < result.data.size(); i++) {
            result.data[i] = in[i];
        }

        return result;
    }

    void resize(const size_t& rows, const size_t& cols) {
        data.resize(rows * cols);
        n_rows = rows;
        n_cols = cols;
    }
};

// The functions below write into an existing out of the right shape, so
// buffers can be reused across calls. out must not be one of the inputs,
// which is what lets the kernels use __restrict. Two Matrix objects never
// share storage, so comparing addresses is enough.
inline void check_out(const Matrix& out, size_t rows, size_t cols, std::initializer_list<const Matrix*> inputs) {
    if (out.rows() != rows || out.cols() != cols) {
        throw std::invalid_argument("Output matrix has the wrong shape.");
    }
    for (const Matrix* in : inputs) {
        if (in == &out) {
            throw std::invalid_argument("Output matrix must not be an input.");
        }
    }
}

// Each call runs on one thread or on all omp_get_max_threads() of them: all
// if at least two threads get the minimum amount of work and there are at
// least two pieces to split, else one, without starting a team. Counts in
// between are slower, because libgomp ends the extra pool threads whenever a
// team shrinks and starts them again for the next bigger one
// (benchmarks/results/D6-compare.json).
//
// Work is multiply-adds for the multiplies and elements for the elementwise
// ops. The minimums come from sweeps at 8 threads on both benchmark networks
// (benchmarks/results/D6-sweep-*.json); the -D overrides exist for those sweeps.
#ifndef ML_MIN_MADDS_PER_THREAD
#define ML_MIN_MADDS_PER_THREAD 5000
#endif
#ifndef ML_MIN_ELEMENTS_PER_THREAD
#define ML_MIN_ELEMENTS_PER_THREAD 8192
#endif
#ifndef ML_MIN_GRAD_ROWS_PER_THREAD
#define ML_MIN_GRAD_ROWS_PER_THREAD 1
#endif
inline constexpr size_t min_madds_per_thread = ML_MIN_MADDS_PER_THREAD;
inline constexpr size_t min_elements_per_thread = ML_MIN_ELEMENTS_PER_THREAD;
inline constexpr size_t min_grad_rows_per_thread = ML_MIN_GRAD_ROWS_PER_THREAD;

inline int choose_threads(size_t work, size_t min_work, size_t max_parts = SIZE_MAX) {
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    return ((work / min_work >= 2 && max_parts >= 2) ? max_threads : 1);
}

// The thread count of each kernel for a given shape.
namespace thread_count {

// A·B, A: m×n, B: n×q, split over output rows in 4-row blocks.
inline int matmul(size_t m, size_t n, size_t q) {
    return choose_threads(m * n * q, min_madds_per_thread, m / 4);
}

// Aᵀ·B, A: m×p, B: m×q, split over the p rows of the result.
inline int matmul_tn_outputs(size_t m, size_t p, size_t q) {
    return choose_threads(m * p * q, min_madds_per_thread, p / min_grad_rows_per_thread);
}

// Aᵀ·B split over the m rows of the sum, in 4-row blocks.
inline int matmul_tn_rows(size_t m, size_t p, size_t q) {
    return choose_threads(m * p * q, min_madds_per_thread, m / 4);
}

inline int elementwise(size_t n) {
    return choose_threads(n, min_elements_per_thread);
}

// Column sums of an m×n matrix, split over the columns.
inline int column_sums(size_t m, size_t n) {
    return choose_threads(m * n, min_elements_per_thread, n);
}

}

// One scratch buffer per thread for reductions. Each is its own 64-byte
// aligned allocation, rounded up to whole cache lines, so threads never write
// to the same line. Buffers only grow, so training allocates nothing after
// the first epoch.
class ThreadPartials {
private:
    struct AlignedFree {
        void operator()(double* p) const {
            ::operator delete[](p, std::align_val_t(64));
        }
    };

    std::vector<std::unique_ptr<double[], AlignedFree>> bufs;
    size_t cap = 0; // doubles per buffer

public:
    ThreadPartials() = default;

    // Scratch only, so a copy starts empty.
    ThreadPartials(const ThreadPartials&) {}
    ThreadPartials& operator=(const ThreadPartials&) {
        return *this;
    }

    // At least threads buffers of at least n doubles.
    void ensure(size_t threads, size_t n) {
        if (n > cap) {
            bufs.clear();
            cap = (n + 7) / 8 * 8;
        }
        while (bufs.size() < threads) {
            void* p = ::operator new[](cap * sizeof(double), std::align_val_t(64));
            bufs.emplace_back(static_cast<double*>(p));
        }
    }

    double* get(size_t t) {
        return bufs[t].get();
    }
};

// Without __restrict GCC has to assume the output may overlap an input and
// does not vectorize these loops. It is on the parameters because GCC ignores
// it on local pointers once the function is inlined.
//
// The multiplies work on 4 rows per pass, so each value loaded from b is used
// 4 times (matmul_rows), or each element of c is loaded and stored once per
// 4 terms (matmul_tn). Every element of c still gets one += per term, in the
// same order as the one-row loop, so results are bit-identical.
namespace kernels {

// Rows c0..c3 += x0..x3 * b. Each row is its own __restrict pointer, since GCC
// cannot prove that rows a runtime stride apart do not overlap.
inline void axpy4(const double* __restrict b, double x0, double x1, double x2, double x3,
                  double* __restrict c0, double* __restrict c1, double* __restrict c2, double* __restrict c3,
                  size_t q) {
    for (size_t j = 0; j < q; j++) {
        double bj = b[j];
        c0[j] += x0 * bj;
        c1[j] += x1 * bj;
        c2[j] += x2 * bj;
        c3[j] += x3 * bj;
    }
}

inline void axpy(const double* __restrict b, double x, double* __restrict c, size_t q) {
    for (size_t j = 0; j < q; j++) {
        c[j] += x * b[j];
    }
}

// This thread's contiguous share [begin, end) of n items. Call inside a
// parallel region; without OpenMP it is [0, n).
inline void thread_range(size_t n, size_t& begin, size_t& end) {
    size_t t = 0, nt = 1;
#ifdef _OPENMP
    t = omp_get_thread_num();
    nt = omp_get_num_threads();
#endif
    begin = n * t / nt;
    end = n * (t + 1) / nt;
}

// Sets rows rows of c (q wide) to row0, or to zero if row0 is null.
inline void init_rows(double* __restrict c, const double* __restrict row0, size_t rows, size_t q) {
    for (size_t r = 0; r < rows; r++) {
        if (row0) {
            std::copy(row0, row0 + q, c + r * q);
        } else {
            std::fill(c + r * q, c + (r + 1) * q, 0.0);
        }
    }
}

// c (m×q) = row0 + a·b, a: m×n, b: n×q, row0: 1×q (null for zeros).
// In i-k-j order every read is contiguous. Each element of c starts at row0
// and adds its terms in k order. Threads split the output rows in 4-row
// blocks and each owns its rows, so results are the same for any thread count.
inline void matmul_rows(const double* __restrict a, const double* __restrict b, const double* __restrict row0,
                        double* __restrict c, size_t m, size_t n, size_t q, int threads) {
    size_t nblocks = m / 4;
    #pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
    for (size_t blk = 0; blk < nblocks; blk++) {
        size_t i = blk * 4;
        const double* a0 = a + i * n;
        double* c0 = c + i * q;
        init_rows(c0, row0, 4, q);
        for (size_t k = 0; k < n; k++) {
            axpy4(b + k * q, a0[k], a0[n + k], a0[2 * n + k], a0[3 * n + k],
                  c0, c0 + q, c0 + 2 * q, c0 + 3 * q, q);
        }
    }
    // Leftover rows, on one thread.
    for (size_t i = nblocks * 4; i < m; i++) {
        init_rows(c + i * q, row0, 1, q);
        for (size_t k = 0; k < n; k++) {
            axpy(b + k * q, a[i * n + k], c + i * q, q);
        }
    }
}

// Rows j0..j1 of c (p×q) += what rows i0..i1 of a (m×p) and b (m×q) add to
// aᵀ·b. Four rows at a time: four += per element of c, in row order, per
// load and store.
inline void matmul_tn_part(const double* __restrict a, const double* __restrict b, double* __restrict c,
                           size_t i0, size_t i1, size_t p, size_t q, size_t j0, size_t j1) {
    size_t i = i0;
    for (; i + 4 <= i1; i += 4) {
        const double* b0 = b + i * q;
        const double* b1 = b0 + q;
        const double* b2 = b1 + q;
        const double* b3 = b2 + q;
        for (size_t j = j0; j < j1; j++) {
            double x0 = a[i * p + j], x1 = a[(i + 1) * p + j];
            double x2 = a[(i + 2) * p + j], x3 = a[(i + 3) * p + j];
            double* cj = c + j * q;
            for (size_t k = 0; k < q; k++) {
                double s = cj[k];
                s += x0 * b0[k];
                s += x1 * b1[k];
                s += x2 * b2[k];
                s += x3 * b3[k];
                cj[k] = s;
            }
        }
    }
    for (; i < i1; i++) {
        for (size_t j = j0; j < j1; j++) {
            axpy(b + i * q, a[i * p + j], c + j * q, q);
        }
    }
}

// c (p×q) += aᵀ·b, a: m×p, b: m×q. Threads split the rows of c and each adds
// all m rows into its own, in order, so results are the same for any thread
// count. Every thread reads all of a and b.
inline void matmul_tn(const double* __restrict a, const double* __restrict b, double* __restrict c,
                      size_t m, size_t p, size_t q, int threads) {
    #pragma omp parallel num_threads(threads) if (threads > 1)
    {
        size_t j0, j1;
        thread_range(p, j0, j1);
        matmul_tn_part(a, b, c, 0, m, p, q, j0, j1);
    }
}

// c (p×q) = aᵀ·b, with the m-row sum split across threads instead. Each
// thread adds its range of 4-row blocks into its own zeroed partial, then the
// partials are summed in thread order. That regroups the additions, so the
// last bits differ from the serial sum, but runs with the same thread count
// agree, and one thread gives the serial result. partials must hold at least
// threads buffers.
inline void matmul_tn_reduce(const double* __restrict a, const double* __restrict b, double* __restrict c,
                             size_t m, size_t p, size_t q, ThreadPartials& partials, int threads) {
    size_t nblocks = m / 4;
    #pragma omp parallel num_threads(threads) if (threads > 1)
    {
        size_t t = 0, nt = 1;
#ifdef _OPENMP
        t = omp_get_thread_num();
        nt = omp_get_num_threads();
#endif
        double* P = partials.get(t);
        std::fill(P, P + p * q, 0.0);
        size_t blk0, blk1;
        thread_range(nblocks, blk0, blk1);
        // The last thread also takes the leftover rows.
        size_t i1 = (t == nt - 1) ? m : blk1 * 4;
        matmul_tn_part(a, b, P, blk0 * 4, i1, p, q, 0, p);

        #pragma omp barrier
        size_t e0, e1;
        thread_range(p * q, e0, e1);
        for (size_t e = e0; e < e1; e++) {
            double s = partials.get(0)[e];
            for (size_t u = 1; u < nt; u++) {
                s += partials.get(u)[e];
            }
            c[e] = s;
        }
    }
}

}

// out = Aᵀ.
inline void transpose(const Matrix& A, Matrix& out) {
    ML_PROFILE_SCOPE("Matrix::transpose_into");
    check_out(out, A.cols(), A.rows(), {&A});

    const double* a = A.raw();
    double* c = out.raw();
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < A.cols(); j++) {
            c[j * A.rows() + i] = a[i * A.cols() + j];
        }
    }
}

// out = A·B, A: m×n, B: n×q, out: m×q.
inline void matmul(const Matrix& A, const Matrix& B, Matrix& out) {
    ML_PROFILE_SCOPE("Matrix::matmul_into");
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions are not valid.");
    }
    check_out(out, A.rows(), B.cols(), {&A, &B});
    ML_PROFILE_FLOPS(2 * A.rows() * A.cols() * B.cols());

    int threads = thread_count::matmul(A.rows(), A.cols(), B.cols());
    ML_PROFILE_THREADS(threads);
    kernels::matmul_rows(A.raw(), B.raw(), nullptr, out.raw(), A.rows(), A.cols(), B.cols(), threads);
}

// out = Aᵀ·B without building Aᵀ, A: m×p, B: m×q, out: p×q.
inline void matmul_tn(const Matrix& A, const Matrix& B, Matrix& out) {
    ML_PROFILE_SCOPE("Matrix::matmul_tn");
    if (A.rows() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions are not valid.");
    }
    check_out(out, A.cols(), B.cols(), {&A, &B});
    ML_PROFILE_FLOPS(2 * A.rows() * A.cols() * B.cols());

    int threads = thread_count::matmul_tn_outputs(A.rows(), A.cols(), B.cols());
    ML_PROFILE_THREADS(threads);
    std::fill(out.raw(), out.raw() + out.size(), 0.0);
    kernels::matmul_tn(A.raw(), B.raw(), out.raw(), A.rows(), A.cols(), B.cols(), threads);
}

// out = Aᵀ·B with the row sum split across threads (kernels::matmul_tn_reduce).
// Repeatable for a given thread count, but not bit-identical to matmul_tn.
inline void matmul_tn_reduce(const Matrix& A, const Matrix& B, Matrix& out, ThreadPartials& partials) {
    ML_PROFILE_SCOPE("Matrix::matmul_tn");
    if (A.rows() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions are not valid.");
    }
    check_out(out, A.cols(), B.cols(), {&A, &B});
    ML_PROFILE_FLOPS(2 * A.rows() * A.cols() * B.cols());

    int threads = thread_count::matmul_tn_rows(A.rows(), A.cols(), B.cols());
    ML_PROFILE_THREADS(threads);
    partials.ensure(threads, out.size());
    kernels::matmul_tn_reduce(A.raw(), B.raw(), out.raw(), A.rows(), A.cols(), B.cols(), partials, threads);
}

// out = A·W + b, b (1×n) added to every row. Each row starts as a copy of b,
// as the first term of the old bias-column product did, then adds A·W in k order.
inline void matmul_bias(const Matrix& A, const Matrix& W, const Matrix& b, Matrix& out) {
    ML_PROFILE_SCOPE("Matrix::matmul_bias");
    if (A.cols() != W.rows() || b.rows() != 1 || b.cols() != W.cols()) {
        throw std::invalid_argument("Matrix dimensions are not valid.");
    }
    check_out(out, A.rows(), W.cols(), {&A, &W, &b});
    ML_PROFILE_FLOPS(2 * A.rows() * A.cols() * W.cols());

    int threads = thread_count::matmul(A.rows(), A.cols(), W.cols());
    ML_PROFILE_THREADS(threads);
    kernels::matmul_rows(A.raw(), W.raw(), b.raw(), out.raw(), A.rows(), A.cols(), W.cols(), threads);
}

// out (1×n) = column sums of A, each in row order. Threads split the columns.
// Sums go into a local acc (up to 256 columns per pass) and out is written
// once: adding into out on every row made threads that share a cache line of
// out fight over it (5x slower than one thread on the small net).
inline void column_sums(const Matrix& A, Matrix& out) {
    ML_PROFILE_SCOPE("Matrix::column_sums");
    check_out(out, 1, A.cols(), {&A});

    const double* a = A.raw();
    double* c = out.raw();
    size_t n = A.cols();
    int threads = thread_count::column_sums(A.rows(), n);
    ML_PROFILE_THREADS(threads);
    #pragma omp parallel num_threads(threads) if (threads > 1)
    {
        size_t j0, j1;
        kernels::thread_range(n, j0, j1);
        constexpr size_t chunk = 256;
        for (size_t g = j0; g < j1; g += chunk) {
            size_t w = std::min(chunk, j1 - g);
            double acc[chunk];
            std::fill(acc, acc + w, 0.0);
            for (size_t i = 0; i < A.rows(); i++) {
                for (size_t j = 0; j < w; j++) {
                    acc[j] += a[i * n + g + j];
                }
            }
            std::copy(acc, acc + w, c + g);
        }
    }
}
