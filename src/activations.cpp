#include "activations.hpp"
#include <cmath>
#include <stdexcept>

namespace {

float sigmoid_scalar(float x) {
    if (x >= 0.0f) {
        float exp_neg_x = std::exp(-x);
        return 1.0f / (1.0f + exp_neg_x);
    }
    float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
}    

}

namespace activations {

Matrix relu(const Matrix& input) {
    Matrix result = input;
    for (std::size_t i = 0; i < result.size(); ++i ) {
        if (result[i] < 0.0f) result[i] = 0.0f;
    }
    return result;
}
Matrix relu_derivative(const Matrix& input) {
    Matrix result = input;
    for (std::size_t i = 0; i < result.size(); ++i ) {
        if (result[i] <= 0.0f) { result[i] = 0.0f; }
        else { result[i] = 1.0f; }
    }
    return result;
}

Matrix leaky_relu(const Matrix& input, float alpha) {
    Matrix result = input;
    for (std::size_t i = 0; i < result.size(); ++i ) {
        if (result[i] < 0.0f) { result[i] = alpha * result[i]; }
    }
    return result;
}

Matrix leaky_relu_derivative(const Matrix& input, float alpha) {
    Matrix result = input;
    for (std::size_t i = 0; i < result.size(); ++i ) {
        if (result[i] > 0.0f) { result[i] = 1; }
        else { result[i] = alpha; }
    }
    return result;
}

Matrix sigmoid(const Matrix& input) {
    Matrix result = input;
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = sigmoid_scalar(result[i]);
    return result;
};

Matrix sigmoid_derivative(const Matrix& input) {
    Matrix result = input;
    for (std::size_t i = 0; i < result.size(); ++i) {
        float sigmoid_value = sigmoid_scalar(result[i]);
        result[i] = sigmoid_value * (1.0f - sigmoid_value);
    }
    return result;
}

Matrix tanh(const Matrix& input) {
    Matrix result = input;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = std::tanh(result[i]);
    }
    return result;
}
Matrix tanh_derivative(const Matrix& input) {
    Matrix result = input;
    for (std::size_t i = 0; i < result.size(); ++i) {
        float tanh_value = std::tanh(result[i]);
        result[i] = 1 - tanh_value * tanh_value;
    }
    return result;
}

Matrix softmax(const Matrix& input) {
    if (input.get_cols() == 0) throw std::invalid_argument("softmax requires at least one col");
    Matrix result = input;

    for (std::size_t row = 0; row < result.get_rows(); ++row) {
        float max_value = result(row, 0);
        for (std::size_t col = 1; col < result.get_cols(); ++col) {
            if (result(row, col) > max_value) {
                max_value = result(row, col);
            }
        }

        float sum_exp = 0.0f;

        for (std::size_t col = 0; col < result.get_cols(); ++col) {
            result(row, col) /= sum_exp;
        }
    }

    return result;
} // elements in a row depend on each other

}


