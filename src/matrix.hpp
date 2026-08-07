#pragma once
#include <vector>

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
        [[nodiscard]] Matrix& operator+=(const Matrix& other);
        [[nodiscard]] Matrix operator-(const Matrix& other) const;
        [[nodiscard]] Matrix& operator-=(const Matrix& other);
        [[nodiscard]] Matrix operator*(float scalar) const;
        [[nodiscard]] Matrix& operator*=(float scalar);

        [[nodiscard]] Matrix hadamard(const Matrix& other) const; // element wise matrix multiplication
        [[nodiscard]] Matrix matmul(const Matrix& other) const; // og matrix multiplication
        [[nodiscard]] Matrix transpose() const;

        std::size_t get_rows() const;
        std::size_t get_cols() const;
};