#include "matrix.hh"

//___________CONSTRUCTORS__________
Matrix::Matrix(int m, int n) {}


//___________SETTERS__________


//___________GETTERS__________
int Matrix::get_rows() const {
    return this->m;
}

int Matrix::get_cols() const {
    return this->n;
}


Matrix Matrix::matsum(const Matrix& m1, const Matrix& m2) const {}

Matrix Matrix::matsub(const Matrix& m1, const Matrix& m2) const {}

Matrix Matrix::matdiv(const Matrix& m1, const Matrix& m2) const {}

Matrix Matrix::matmul(const Matrix& m1, const Matrix& m2) const {}

Matrix Matrix::matdot(const Matrix& m1, const Matrix& m2) const {}


static void Matrix::print_matrix(const Matrix& m) {}
