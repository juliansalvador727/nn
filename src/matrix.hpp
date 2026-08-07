#pragma once
#include <vector>
#include <cstddef>

class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
public:
    Matrix(std::size_t rows, std::size_t cols);

    ~Matrix() = default;

    Matrix(const Matrix& other) = default;
    Matrix& operator=(const Matrix& other) = default;

    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    // accessing
    float& operator()(std::size_t row, std::size_t col);
    const float& operator()(std::size_t row, std::size_t col) const;

    // basic operations
    [[nodiscard]] Matrix operator+(const Matrix& other) const;
                  Matrix& operator+=(const Matrix& other);
    [[nodiscard]] Matrix operator-(const Matrix& other) const;
                  Matrix& operator-=(const Matrix& other);
    [[nodiscard]] Matrix operator*(float scalar) const;
                  Matrix& operator*=(float scalar);

    [[nodiscard]] Matrix hadamard(const Matrix& other) const; // element wise matrix multiplication
    [[nodiscard]] Matrix matmul(const Matrix& other) const; // og matrix multiplication
    [[nodiscard]] Matrix transpose() const;

    [[nodiscard]] std::size_t get_rows() const noexcept;
    [[nodiscard]] std::size_t get_cols() const noexcept;
};

// for syntactic completeness, allow 2.0f * matrix
[[nodiscard]] Matrix operator*(float scalar, const Matrix& matrix);