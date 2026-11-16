#pragma once
#include <initializer_list>
#include <vector>
#include <cstddef>

class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
public:
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::initializer_list<std::initializer_list<float>> rows);

    ~Matrix() = default;

    Matrix(const Matrix& other) = default;
    Matrix& operator=(const Matrix& other) = default;

    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    // accessing via 2d math
    float& operator()(std::size_t row, std::size_t col);
    const float& operator()(std::size_t row, std::size_t col) const;

    // accessing via flat indexing
    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;

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

    [[nodiscard]] Matrix add_row_vector(const Matrix& row) const;
    [[nodiscard]] Matrix sum_rows() const;

    template <typename F>
    [[nodiscard]] Matrix apply(F fn) const {
        Matrix result = *this;
        for (auto& v: result.data_ ) v = fn(v);
        return result;
    }

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t get_rows() const noexcept;
    [[nodiscard]] std::size_t get_cols() const noexcept;

    // these are for init methods
    [[nodiscard]] static Matrix random_normal(std::size_t rows, std::size_t cols, float mean, float stddev);
    [[nodiscard]] static Matrix xavier(std::size_t fan_in, std::size_t fan_out); // meant for tanh and sigmoid.
    [[nodiscard]] static Matrix he(std::size_t fan_in, std::size_t fan_out); // for relu


};

// for syntactic completeness, allow 2.0f * matrix
[[nodiscard]] Matrix operator*(float scalar, const Matrix& matrix);
