#pragma once
#include "matrix.hpp"

namespace activations 
{
// relu, leaky_relu, sigmoid, tanh, softmax
// d/dx of relu, leaky_relu, sigmoid, tanh, softmax

[[nodiscard]] Matrix relu(const Matrix& input);
[[nodiscard]] Matrix relu_derivative(const Matrix& input);

[[nodiscard]] Matrix leaky_relu(const Matrix& input, float alpha = 0.01f);
[[nodiscard]] Matrix leaky_relu_derivative(const Matrix& input, float alpha = 0.01f);

[[nodiscard]] Matrix sigmoid(const Matrix& input);
[[nodiscard]] Matrix sigmoid_derivative(const Matrix& input);

[[nodiscard]] Matrix tanh(const Matrix& input);
[[nodiscard]] Matrix tanh_derivative(const Matrix& input);

[[nodiscard]] Matrix softmax(const Matrix& input); // elements in a row depend on each other
// todo: add GELU, ELU, Swish

}