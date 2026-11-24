#include <cmath>
#include <iostream>
#include "cpp_ml/models/LinearRegression.hpp"

int main() {
    // Learn y = 1 + 2x from a tiny, self-contained dataset.
    Matrix X(5, 1), y(5, 1);
    for (size_t i = 0; i < X.rows(); ++i) {
        X(i, 0) = static_cast<double>(i) - 2.0;
        y(i, 0) = 1.0 + 2.0 * X(i, 0);
    }

    LinearRegression model;
    model.fit(X, y, 0.1, 1000);
    Matrix query(1, 1);
    query(0, 0) = 3.0;
    const double prediction = model.predict(query)(0, 0);
    std::cout << "Prediction for x = 3: " << prediction << " (expected 7)\n";
    return std::isfinite(prediction) && std::abs(prediction - 7.0) < 1e-6 ? 0 : 1;
}
