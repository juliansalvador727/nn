#pragma once
#include "matrix.hpp"

namespace losses {

[[nodiscard]] float mse(const Matrix& predictions, const Matrix& targets);
[[nodiscard]] Matrix mse_derivative(const Matrix& predictions, const Matrix& targets);
[[nodiscard]] float cross_entropy(const Matrix& probabilities, const std::vector<std::size_t>& labels);

}
