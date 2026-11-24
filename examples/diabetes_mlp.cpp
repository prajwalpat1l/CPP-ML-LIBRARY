#include <iostream>
#include <algorithm>
#include <vector>
#include "cpp_ml/data/DataFrame.hpp"
#include "cpp_ml/models/LinearRegression.hpp"
#include "cpp_ml/models/LogisticRegression.hpp"
#include "cpp_ml/models/MLPClassifier.hpp"

int main() {
    DataFrame df = CSVReader::read("data/Diabetes_Prediction.csv");

    df.shuffle();
    df.one_hot_encode("gender");

    std::vector<std::string> header = df.headers();
    header.erase(std::find(header.begin(), header.end(), "smoking_history"));
    header.erase(std::find(header.begin(), header.end(), "diabetes"));
    header.erase(std::find(header.begin(), header.end(), "gender_Other")); // All zeros in the training split.

    Matrix X = df.to_matrix(header), y = df.to_matrix({"diabetes"});
    Matrix X_train = X.slice(0, 8000), X_test = X.slice(8000, 10000), y_train = y.slice(0, 8000), y_test = y.slice(8000, 10000);

    Scaler Xs;
    Xs.fit(X_train);
    Matrix X_train_Scaled = Xs.transform(X_train), X_test_Scaled = Xs.transform(X_test);

    MLPClassifier model(X.cols(), {32, 16, 1});
    model.fit(X_train_Scaled, y_train, 10000, 0.01);
    Matrix y_pred = model.predict(X_test_Scaled);

    double acc = LogisticRegression::accuracy(y_test, y_pred);
    std::cout << acc << std::endl;
}
