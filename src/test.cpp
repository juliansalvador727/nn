#include "activations.hpp"
#include "loss.hpp"
#include "matrix.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_near(float actual, float expected, float tolerance, const std::string& message) {
    check(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance, message);
}

template <typename Function>
void check_invalid_argument(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_matrix_multiplication() {
    Matrix left{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix right{{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}};

    check(left.get_rows() == 2 && left.get_cols() == 3, "initializer shape");
    check_near(left(1, 2), 6.0f, 1e-6f, "initializer row-major values");
    check_invalid_argument([] { (void)Matrix{{1.0f, 2.0f}, {3.0f}}; },
                           "initializer must reject uneven rows");

    Matrix product = left.matmul(right);
    check(product.get_rows() == 2 && product.get_cols() == 2, "matmul shape");
    check_near(product(0, 0), 58.0f, 1e-5f, "matmul (0, 0)");
    check_near(product(0, 1), 64.0f, 1e-5f, "matmul (0, 1)");
    check_near(product(1, 0), 139.0f, 1e-5f, "matmul (1, 0)");
    check_near(product(1, 1), 154.0f, 1e-5f, "matmul (1, 1)");
    check_invalid_argument([&] { (void)left.matmul(left); }, "matmul must reject incompatible shapes");
}

void test_softmax_and_cross_entropy() {
    Matrix logits(2, 3);
    logits(0, 0) = 1000.0f; logits(0, 1) = 1000.0f; logits(0, 2) = 1000.0f;
    logits(1, 0) = 1.0f; logits(1, 1) = 2.0f; logits(1, 2) = 3.0f;

    Matrix probabilities = activations::softmax(logits);
    for (std::size_t row = 0; row < probabilities.get_rows(); ++row) {
        float row_sum = 0.0f;
        for (std::size_t col = 0; col < probabilities.get_cols(); ++col) {
            check(std::isfinite(probabilities(row, col)), "softmax must be finite");
            check(probabilities(row, col) >= 0.0f, "softmax must be nonnegative");
            row_sum += probabilities(row, col);
        }
        check_near(row_sum, 1.0f, 1e-6f, "softmax row sum");
    }
    check_near(probabilities(0, 0), 1.0f / 3.0f, 1e-6f, "softmax large equal logits");
    check(probabilities(1, 2) > probabilities(1, 1) &&
          probabilities(1, 1) > probabilities(1, 0), "softmax class ordering");

    Matrix known_probabilities(2, 3);
    known_probabilities(0, 0) = 0.1f; known_probabilities(0, 1) = 0.7f;
    known_probabilities(0, 2) = 0.2f; known_probabilities(1, 0) = 0.8f;
    known_probabilities(1, 1) = 0.1f; known_probabilities(1, 2) = 0.1f;
    const std::vector<std::size_t> labels{1, 0};
    const float expected = (-std::log(0.7f) - std::log(0.8f)) / 2.0f;
    check_near(losses::cross_entropy(known_probabilities, labels), expected, 1e-6f,
               "cross-entropy mean loss");

    check_invalid_argument([&] { (void)losses::cross_entropy(known_probabilities, {1}); },
                           "cross-entropy must reject missing labels");
    check_invalid_argument([&] { (void)losses::cross_entropy(known_probabilities, {1, 3}); },
                           "cross-entropy must reject out-of-range labels");
    check_invalid_argument([&] { (void)losses::cross_entropy(Matrix(0, 3), {}); },
                           "cross-entropy must reject empty batches");
    check_invalid_argument([&] { (void)losses::cross_entropy(Matrix(1, 0), {0}); },
                           "cross-entropy must reject zero classes");
}

void test_softmax_cross_entropy_gradient() {
    Matrix logits(2, 3);
    logits(0, 0) = 0.2f; logits(0, 1) = -0.4f; logits(0, 2) = 1.1f;
    logits(1, 0) = -0.3f; logits(1, 1) = 0.8f; logits(1, 2) = 0.1f;
    const std::vector<std::size_t> labels{2, 1};
    const Matrix probabilities = activations::softmax(logits);
    const Matrix gradient = losses::softmax_cross_entropy_gradient(probabilities, labels);

    constexpr float step = 1e-3f;
    for (std::size_t row = 0; row < logits.get_rows(); ++row) {
        for (std::size_t col = 0; col < logits.get_cols(); ++col) {
            Matrix plus = logits;
            Matrix minus = logits;
            plus(row, col) += step;
            minus(row, col) -= step;
            const float loss_plus = losses::cross_entropy(activations::softmax(plus), labels);
            const float loss_minus = losses::cross_entropy(activations::softmax(minus), labels);
            const float numerical = (loss_plus - loss_minus) / (2.0f * step);
            check_near(gradient(row, col), numerical, 2e-4f,
                       "softmax cross-entropy gradient at (" + std::to_string(row) + ", " +
                           std::to_string(col) + ")");
        }
    }

    check_invalid_argument([&] { (void)losses::softmax_cross_entropy_gradient(probabilities, {2}); },
                           "gradient must reject missing labels");
    check_invalid_argument([&] { (void)losses::softmax_cross_entropy_gradient(probabilities, {2, 3}); },
                           "gradient must reject out-of-range labels");
    check_invalid_argument([&] { (void)losses::softmax_cross_entropy_gradient(Matrix(0, 3), {}); },
                           "gradient must reject empty batches");
}

void test_mse() {
    Matrix predictions(1, 2);
    predictions(0, 0) = 2.0f; predictions(0, 1) = 4.0f;
    Matrix targets(1, 2);
    targets(0, 0) = 1.0f; targets(0, 1) = 2.0f;

    check_near(losses::mse(predictions, targets), 2.5f, 1e-6f, "MSE value");
    Matrix gradient = losses::mse_derivative(predictions, targets);
    check_near(gradient(0, 0), 1.0f, 1e-6f, "MSE gradient (0, 0)");
    check_near(gradient(0, 1), 2.0f, 1e-6f, "MSE gradient (0, 1)");
    check_invalid_argument([&] { (void)losses::mse(predictions, Matrix(1, 3)); },
                           "MSE must reject mismatched shapes");
}

} // namespace

int main() {
    try {
        test_matrix_multiplication();
        test_softmax_and_cross_entropy();
        test_softmax_cross_entropy_gradient();
        test_mse();
        std::cout << "All tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
