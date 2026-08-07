#include <vector>

class Matrix {
    private:
        int rows;
        int cols;
        std::vector<float> data;
    public:
        Matrix(int rows, int cols);

        // accessing
        float& operator()(int row, int col);
        const float& operator()(int row, int col) const;

        // basic operations
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(float scalr) const;

        Matrix hadamard(const Matrix& other) const;
        Matrix matmul(const Matrix& other) const;
        Matrix transpose() const;

        int get_rows() const;
        int get_cols() const;
};