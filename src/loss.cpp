#include "loss.hpp"
#include <cstddef>
#include <stdexcept>
#include <cmath>

namespace losses {
    float mse(const Matrix& predictions, const Matrix& targets) {
        if (predictions.get_rows() != targets.get_rows() || predictions.get_cols() != predictions.get_cols()) throw std::invalid_argument("predictions and targets must have same dim");
        if (predictions.size() == 0) throw std::invalid_argument("cannot calculate MSE of empty matrices");
        float result = 0.0f;
        for (std::size_t i = 0; i < predictions.size(); ++i) {
            float difference = (predictions[i] - targets[i]);
            result += difference * difference;
        }
        return result / static_cast<float>(predictions.size());
    }
    Matrix mse_derivative(const Matrix& predictions, const Matrix& targets) {
        if (predictions.get_rows() != targets.get_rows() || predictions.get_cols() != predictions.get_cols()) throw std::invalid_argument("predictions and targets must have same dim");
        if (predictions.size() == 0) throw std::invalid_argument("cannot calculate MSE of empty matrices");
        Matrix result = predictions - targets;
        result *= 2.0f / static_cast<float>(predictions.size());
        return result;
    }
    float cross_entropy(const Matrix& predictions, const Matrix& targets) {
        if (predictions.get_rows() != targets.get_rows() || predictions.get_cols() != predictions.get_cols()) throw std::invalid_argument("predictions and targets must have same dim");
        if (predictions.size() == 0) throw std::invalid_argument("cannot calculate MSE of empty matrices");
        constexpr float epsilon = 1e-7f;
        float total_loss = 0.0f;
        for (std::size_t row = 0; row < predictions.get_rows(); ++row) {
            for (std::size_t col = 0; col < predictions.get_cols(); ++col) {
                float probability = predictions(row, col);

                if (probability < epsilon) probability = epsilon; 

                total_loss -= targets(row, col) * std::log(probability);
            }
        }
        return total_loss / static_cast<float>(predictions.get_rows());
    }
}