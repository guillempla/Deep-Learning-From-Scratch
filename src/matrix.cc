#include "matrix.hh"
using namespace std;

//___________CONSTRUCTORS__________
Matrix::Matrix(unsigned m, unsigned n) {
    this->m = m;
    this->n = n;
    this->m_matrix.resize(m);
    for (int i = 0; i < m; i++)
        m_matrix[i].resize(n, 0);
}

Matrix::Matrix(unsigned m, unsigned n, double initial) {
    this->m = m;
    this->n = n;
    this->m_matrix.resize(m);
    for (int i = 0; i < m; i++)
        m_matrix[i].resize(n, initial);
}

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


void Matrix::print_matrix() const {}
