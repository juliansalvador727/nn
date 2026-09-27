#include "loss.hpp"
#include <cstddef>
#include <stdexcept>
#include <cmath>

namespace losses {
    float mse(const Matrix& predictions, const Matrix& targets) {
        if (predictions.get_rows() != targets.get_rows() || predictions.get_cols() != targets.get_cols()) throw std::invalid_argument("predictions and targets must have same dim");
        if (predictions.size() == 0) throw std::invalid_argument("cannot calculate MSE of empty matrices");
        float result = 0.0f;
        for (std::size_t i = 0; i < predictions.size(); ++i) {
            float difference = (predictions[i] - targets[i]);
            result += difference * difference;
        }
        return result / static_cast<float>(predictions.size());
    }
    Matrix mse_derivative(const Matrix& predictions, const Matrix& targets) {
        if (predictions.get_rows() != targets.get_rows() || predictions.get_cols() != targets.get_cols()) throw std::invalid_argument("predictions and targets must have same dim");
        if (predictions.size() == 0) throw std::invalid_argument("cannot calculate MSE of empty matrices");
        Matrix result = predictions - targets;
        result *= 2.0f / static_cast<float>(predictions.size());
        return result;
    }
    // float cross_entropy(const Matrix& predictions, const Matrix& targets) {
    //     if (predictions.get_rows() != targets.get_rows() || predictions.get_cols() != predictions.get_cols()) throw std::invalid_argument("predictions and targets must have same dim");
    //     if (predictions.size() == 0) throw std::invalid_argument("cannot calculate MSE of empty matrices");
    //     constexpr float epsilon = 1e-7f;
    //     float total_loss = 0.0f;
    //     for (std::size_t row = 0; row < predictions.get_rows(); ++row) {
    //         for (std::size_t col = 0; col < predictions.get_cols(); ++col) {
    //             float probability = predictions(row, col);

    //             if (probability < epsilon) probability = epsilon; 

    //             total_loss -= targets(row, col) * std::log(probability);
    //         }
    //     }
    //     return total_loss / static_cast<float>(predictions.get_rows());
    // }
    float cross_entropy(const Matrix& predictions, const std::vector<std::size_t>& labels) {
        if (predictions.get_rows() == 0 || predictions.get_cols() == 0) throw std::invalid_argument("predictions must have rows and columns");
        if (labels.size() != predictions.get_rows()) throw std::invalid_argument("# of prediction rows != # of label rows");
        constexpr float epsilon = 1e-7f;
        float total_loss = 0.0f;
        for (std::size_t row = 0; row < predictions.get_rows(); ++row) {
            if (labels[row] >= predictions.get_cols()) {
                throw std::invalid_argument("label is outside class range");
            }
            float probability = predictions(row, labels[row]);
            if (probability < epsilon) probability = epsilon;
            total_loss -= std::log(probability);
        }
        return (1.0f / predictions.get_rows()) * total_loss;
    }
    
    Matrix softmax_cross_entropy_gradient(const Matrix &probabilities, const std::vector<std::size_t> &labels) {
        if (probabilities.get_rows() == 0 || probabilities.get_cols() == 0) throw std::invalid_argument("probabilities must have rows and columns");
        if (labels.size() != probabilities.get_rows()) throw std::invalid_argument("# of prediction rows != # of label rows");
        Matrix result = probabilities;
        for (std::size_t row = 0; row < result.get_rows(); ++row) {
            if (labels[row] >= probabilities.get_cols()) {
                throw std::invalid_argument("label is outside class range");
            }
            result(row, labels[row]) -= 1.0f;
        }

        result *= 1.0f / static_cast<float>(result.get_rows());
        return result;
    }
}