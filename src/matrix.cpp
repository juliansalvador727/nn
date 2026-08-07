#include <vector>
#include "matrix.hpp"

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {};

float& Matrix::operator()(int row, int col) { return data_[row * cols_ + col]; }

const float& Matrix::operator()(int row, int col) const { return data_[row * cols_ + col]; }

Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result = *this; 
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(i,j) += other(i,j);
        }
    }
    return result;
};

Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result = *this;
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(i, j) -= other(i, j); 
        }
    }
    return result;
}
Matrix Matrix::operator*(float scalar) const {
    Matrix result = *this;
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(i,j) *= scalar;
        }
    }
    return result;
}
