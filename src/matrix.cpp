#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include "matrix.hpp"

Matrix::Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> rows)
    : rows_(rows.size()), cols_(rows.size() == 0 ? 0 : rows.begin()->size()) {
    for (const auto& row : rows) {
        if (row.size() != cols_) throw std::invalid_argument("matrix rows must have equal lengths");
    }

    data_.reserve(rows_ * cols_);
    for (const auto& row : rows) {
        data_.insert(data_.end(), row.begin(), row.end());
    }
}

float& Matrix::operator()(std::size_t row, std::size_t col) { return data_[row * cols_ + col]; }
const float& Matrix::operator()(std::size_t row, std::size_t col) const { return data_[row * cols_ + col]; }

float& Matrix::operator[](std::size_t index) { return data_[index]; }
const float& Matrix::operator[](std::size_t index) const { return data_[index]; }

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

std::size_t Matrix::size() const noexcept {
    return data_.size();
}

std::size_t Matrix::get_rows() const noexcept{
    return rows_;
}

std::size_t Matrix::get_cols() const noexcept{
    return cols_;
}

Matrix Matrix::add_row_vector(const Matrix& row) const {
    if (row.rows_ != 1 || row.cols_ != cols_) throw std::invalid_argument("row vector must be 1 x cols for broadcast add");
    Matrix result = *this;
    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            result(r, c) += row(0, c);
        }
    }
    return result;
}

Matrix Matrix::sum_rows() const {
    Matrix result(1, cols_);
    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            result(0, c) += (*this)(r, c);
        }
    }
    return result;
}

namespace {

std::mt19937& rng() {
    static std::mt19937 engine{std::random_device{}()};
    return engine;
}

}

Matrix Matrix::random_normal(std::size_t rows, std::size_t cols, float mean, float stddev) {
    if (!(stddev > 0.0f)) throw std::invalid_argument("stddev must be positive");
    std::normal_distribution<float> dist(mean, stddev);
    Matrix result(rows, cols);
    for (auto& value : result.data_) value = dist(rng());
    return result;
}

Matrix Matrix::xavier(std::size_t fan_in, std::size_t fan_out) {
    if (fan_in == 0) throw std::invalid_argument("fan_in must be nonzero");
    return random_normal(fan_in, fan_out, 0.0f, std::sqrt(1.0f / static_cast<float>(fan_in)));
}

Matrix Matrix::he(std::size_t fan_in, std::size_t fan_out) {
    if (fan_in == 0) throw std::invalid_argument("fan_in must be nonzero");
    return random_normal(fan_in, fan_out, 0.0f, std::sqrt(2.0f / static_cast<float>(fan_in)));
}
