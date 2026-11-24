// Times MLPClassifier::fit and prints one JSON object. Only training is timed.
// The fingerprint (predictions after training) checks the results. Built as
// bench_mlp for timing and bench_mlp_prof with ML_PROFILE for a per-op breakdown.
//
// Usage: bench_mlp [--hidden 32,16] [--epochs N] [--lr 0.01] [--seed 42]
//                  [--data path] [--train-rows 8000] [--test-rows 2000]
//                  [--wgrad outputs|rows]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cpp_ml/data/DataFrame.hpp"
#include "cpp_ml/models/LinearRegression.hpp"
#include "cpp_ml/models/LogisticRegression.hpp"
#include "cpp_ml/models/MLPClassifier.hpp"
#include "cpp_ml/core/Profiler.hpp"

namespace {

struct Options {
    std::vector<size_t> hidden = {32, 16};
    size_t epochs = 200;
    double lr = 0.01;
    std::uint32_t seed = 42;
    std::string data = "data/Diabetes_Prediction.csv";
    size_t train_rows = 8000;
    size_t test_rows = 2000;
    std::string wgrad = "outputs";
};

std::vector<size_t> parse_sizes(const std::string& s) {
    std::vector<size_t> result;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) {
            comma = s.size();
        }
        result.push_back(std::stoul(s.substr(pos, comma - pos)));
        pos = comma + 1;
    }
    return result;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + arg);
        }
        std::string val = argv[++i];
        if (arg == "--hidden") {
            opt.hidden = parse_sizes(val);
        } else if (arg == "--epochs") {
            opt.epochs = std::stoul(val);
        } else if (arg == "--lr") {
            opt.lr = std::stod(val);
        } else if (arg == "--seed") {
            opt.seed = static_cast<std::uint32_t>(std::stoul(val));
        } else if (arg == "--data") {
            opt.data = val;
        } else if (arg == "--train-rows") {
            opt.train_rows = std::stoul(val);
        } else if (arg == "--test-rows") {
            opt.test_rows = std::stoul(val);
        } else if (arg == "--wgrad") {
            if (val != "outputs" && val != "rows") {
                throw std::invalid_argument("--wgrad must be outputs or rows.");
            }
            opt.wgrad = val;
        } else {
            throw std::invalid_argument("Unknown argument " + arg);
        }
    }
    return opt;
}

double log_loss(const Matrix& y, const Matrix& p) {
    double eps = 1e-15;
    double sum = 0;
    for (size_t i = 0; i < y.rows(); i++) {
        double q = std::clamp(p(i, 0), eps, 1 - eps);
        sum -= y(i, 0) * std::log(q) + (1 - y(i, 0)) * std::log(1 - q);
    }
    return sum / static_cast<double>(y.rows());
}

// FNV-1a over the bytes of the predictions: equal hashes mean identical bits.
std::string fnv1a_hex(const Matrix& p) {
    std::uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < p.rows(); i++) {
        for (size_t j = 0; j < p.cols(); j++) {
            double v = p(i, j);
            unsigned char bytes[sizeof(double)];
            std::memcpy(bytes, &v, sizeof(double));
            for (unsigned char b : bytes) {
                h = (h ^ b) * 1099511628211ull;
            }
        }
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// Under perf stat --control=fifo:CTL,ACK (run_bench.py --perf), ML_PERF_CTL and
// ML_PERF_ACK name the FIFOs. Counting is switched on only around fit().
class PerfControl {
private:
    FILE* ctl = nullptr;
    FILE* ack = nullptr;

public:
    PerfControl() {
        const char* c = std::getenv("ML_PERF_CTL");
        const char* a = std::getenv("ML_PERF_ACK");
        if (c && a) {
            ctl = std::fopen(c, "w");
            ack = std::fopen(a, "r");
        }
    }

    ~PerfControl() {
        if (ctl) {
            std::fclose(ctl);
        }
        if (ack) {
            std::fclose(ack);
        }
    }

    // Sends "enable" or "disable" and waits for perf to acknowledge it.
    void send(const char* cmd) {
        if (!ctl || !ack) {
            return;
        }
        std::fprintf(ctl, "%s\n", cmd);
        std::fflush(ctl);
        char buf[16];
        if (!std::fgets(buf, sizeof(buf), ack)) {
            std::cerr << "bench_mlp: no ack from perf" << std::endl;
        }
    }
};

std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

}

int main(int argc, char** argv) {
    Options opt;
    try {
        opt = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "bench_mlp: " << e.what() << std::endl;
        return 2;
    }

    // Same preprocessing as main.cpp.
    DataFrame df = CSVReader::read(opt.data);
    df.shuffle();
    df.one_hot_encode("gender");

    std::vector<std::string> header = df.headers();
    for (const char* drop : {"smoking_history", "diabetes", "gender_Other"}) {
        header.erase(std::find(header.begin(), header.end(), drop));
    }

    size_t train_end = opt.train_rows, test_end = opt.train_rows + opt.test_rows;
    Matrix X = df.to_matrix(header), y = df.to_matrix({"diabetes"});
    Matrix X_train = X.slice(0, train_end), X_test = X.slice(train_end, test_end);
    Matrix y_train = y.slice(0, train_end), y_test = y.slice(train_end, test_end);

    Scaler Xs;
    Xs.fit(X_train);
    Matrix X_train_scaled = Xs.transform(X_train), X_test_scaled = Xs.transform(X_test);

    std::vector<size_t> layers = opt.hidden;
    layers.push_back(1);
    MLPClassifier model(X.cols(), layers, opt.seed);
    model.set_grad_split((opt.wgrad == "rows") ? GradSplit::rows : GradSplit::outputs);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    PerfControl perf;
#ifdef ML_PROFILE
    profiler::reset();
#endif
    perf.send("enable");
    auto start = std::chrono::steady_clock::now();
    model.fit(X_train_scaled, y_train, opt.epochs, opt.lr);
    auto end = std::chrono::steady_clock::now();
    perf.send("disable");
    double train_seconds = std::chrono::duration<double>(end - start).count();
#ifdef ML_PROFILE
    std::vector<profiler::Entry> profile = profiler::entries();
#endif

    Matrix p_train = model.predict(X_train_scaled);
    Matrix p_test = model.predict(X_test_scaled);
    double pred_sum = 0;
    for (size_t i = 0; i < p_test.rows(); i++) {
        pred_sum += p_test(i, 0);
    }

    std::string hidden_json;
    for (size_t i = 0; i < opt.hidden.size(); i++) {
        hidden_json += (i ? "," : "") + std::to_string(opt.hidden[i]);
    }
    std::string head_json;
    for (size_t i = 0; i < std::min<size_t>(5, p_test.rows()); i++) {
        head_json += (i ? ", " : "") + num(p_test(i, 0));
    }

    std::cout << "{\n"
              << "  \"hidden\": [" << hidden_json << "],\n"
              << "  \"epochs\": " << opt.epochs << ",\n"
              << "  \"lr\": " << num(opt.lr) << ",\n"
              << "  \"seed\": " << opt.seed << ",\n"
              << "  \"threads\": " << threads << ",\n"
              << "  \"wgrad\": \"" << opt.wgrad << "\",\n"
              << "  \"train_rows\": " << X_train.rows() << ",\n"
              << "  \"features\": " << X.cols() << ",\n"
              << "  \"profiled\": " << (profiler::enabled ? "true" : "false") << ",\n"
              << "  \"train_seconds\": " << num(train_seconds) << ",\n"
              << "  \"fingerprint\": {\n"
              << "    \"test_accuracy\": " << num(LogisticRegression::accuracy(y_test, p_test)) << ",\n"
              << "    \"train_log_loss\": " << num(log_loss(y_train, p_train)) << ",\n"
              << "    \"test_log_loss\": " << num(log_loss(y_test, p_test)) << ",\n"
              << "    \"test_pred_sum\": " << num(pred_sum) << ",\n"
              << "    \"test_pred_head\": [" << head_json << "],\n"
              << "    \"test_pred_hash\": \"" << fnv1a_hex(p_test) << "\"\n"
              << "  }";
#ifdef ML_PROFILE
    std::cout << ",\n  \"profile\": [";
    for (size_t i = 0; i < profile.size(); i++) {
        const profiler::Entry& e = profile[i];
        std::cout << (i ? "," : "") << "\n    {\"name\": \"" << e.name << "\", \"calls\": " << e.calls
                  << ", \"total_seconds\": " << num(e.total_ns * 1e-9)
                  << ", \"self_seconds\": " << num(e.self_ns * 1e-9)
                  << ", \"flops\": " << e.flops << ", \"threads\": {";
        bool first = true;
        for (const auto& [t, calls] : e.thread_calls) {
            std::cout << (first ? "" : ", ") << "\"" << t << "\": " << calls;
            first = false;
        }
        std::cout << "}}";
    }
    std::cout << "\n  ]";
#endif
    std::cout << "\n}" << std::endl;
}
