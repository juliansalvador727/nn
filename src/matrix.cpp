#include <stdexcept>
#include <vector>
#include "matrix.hpp"

Matrix::Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

float& Matrix::operator()(std::size_t row, std::size_t col) { return data_[row * cols_ + col]; }

const float& Matrix::operator()(std::size_t row, std::size_t col) const { return data_[row * cols_ + col]; }

Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result = *this;
    result += other;
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) throw std::invalid_argument("invalid dim for matrix add");
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i];
    return *this;
}

Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result = *this;
    result -= other;
    return result;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) throw std::invalid_argument("invalid dim for matrix sub");
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] -= other.data_[i];
    return *this;
}

Matrix operator*(float scalar, const Matrix& matrix) {
    return matrix * scalar;
}

Matrix Matrix::operator*(float scalar) const {
    Matrix result = *this;
    result *= scalar;
    return result;
}

Matrix& Matrix::operator*=(float scalar) {
    for (auto& value : data_) value *= scalar;
    return *this;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) throw std::invalid_argument("invalid dim for matrix sub");
    Matrix result = *this;
    for (std::size_t i = 0; i < data_.size(); ++i) result.data_[i] *= other.data_[i];
    return result;
}

Matrix Matrix::matmul(const Matrix& other) const {
    if (cols_ != other.rows_) throw std::invalid_argument("invalid dim for matmul");

    Matrix result(rows_, other.cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < other.cols_; ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < cols_; ++k) {
                sum += (*this)(i, k) * other(k,j); // other matrix  jumps through rows need to optimize later.
            }

            result(i, j) = sum;
        }
    }
    return result;
}
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i,j);
        }
    }
    return result;
}   

std::size_t Matrix::get_rows() const noexcept{
    return rows_;
}

std::size_t Matrix::get_cols() const noexcept{
    return cols_;
}