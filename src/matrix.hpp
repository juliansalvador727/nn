#pragma once
#include <vector>

class Matrix {
    private:
        int rows_;
        int cols_;
        std::vector<float> data_;
    public:
        Matrix(int rows, int cols);

        ~Matrix() = default;

        Matrix(const Matrix& other) = default;
        Matrix& operator=(const Matrix& other) = default;

        Matrix(Matrix&& other) noexcept = default;
        Matrix& operator=(Matrix&& other) = default;

        // accessing
        float& operator()(int row, int col);
        const float& operator()(int row, int col) const;

        // basic operations
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(float scalr) const;

        Matrix hadamard(const Matrix& other) const; // element wise matrix multiplication
        Matrix matmul(const Matrix& other) const; // og matrix multiplication
        Matrix transpose() const;

        int get_rows() const;
        int get_cols() const;
};