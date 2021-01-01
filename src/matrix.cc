#include "matrix.hh"
#include <stdexcept>

using namespace std;

//___________CONSTRUCTORS__________
Matrix::Matrix() {}

Matrix::Matrix(unsigned m, unsigned n) {
    m_rowSize = m;
    m_colSize = n;
    this->m_matrix.resize(m);
    for (int i = 0; i < m; i++)
        m_matrix[i].resize(n, 0);
}

Matrix::Matrix(unsigned m, unsigned n, double initial) {
    m_rowSize = m;
    m_colSize = n;
    this->m_matrix.resize(m);
    for (int i = 0; i < m; i++)
        m_matrix[i].resize(n, initial);
}

//___________SETTERS__________


//___________OPERATIONS__________
// Addition of Two Matrices
Matrix Matrix::operator+(Matrix &B){
    Matrix sum(m_colSize, m_rowSize, 0.0);
    unsigned i,j;
    for (i = 0; i < m_rowSize; i++) {
        for (j = 0; j < m_colSize; j++) {
            sum(i,j) = this->m_matrix[i][j] + B(i,j);
        }
    }
    return sum;
}

// Subtraction of Two Matrices
Matrix Matrix::operator-(Matrix & B){
    Matrix diff(m_colSize, m_rowSize, 0.0);
    unsigned i,j;
    for (i = 0; i < m_rowSize; i++) {
        for (j = 0; j < m_colSize; j++) {
            diff(i,j) = this->m_matrix[i][j] - B(i,j);
        }
    }

    return diff;
}

// Multiplication of Two Matrices
Matrix Matrix::operator*(Matrix & B){
    Matrix multip(m_rowSize, B.getCols(),0.0);
    if (m_colSize == B.getRows()) {
        unsigned i,j,k;
        double temp = 0.0;
        for (i = 0; i < m_rowSize; i++) {
            for (j = 0; j < B.getCols(); j++) {
                temp = 0.0;
                for (k = 0; k < m_colSize; k++) {
                    temp += m_matrix[i][k] * B(k,j);
                }
                multip(i,j) = temp;
                //cout << multip(i,j) << " ";
            }
            //cout << endl;
        }
        return multip;
    }
    else {
        throw std::invalid_argument("ERROR: Wrong matrix dimension!");
    }
}

// Scalar Addition
Matrix Matrix::operator+(double scalar){
    Matrix result(m_rowSize,m_colSize,0.0);
    unsigned i,j;
    for (i = 0; i < m_rowSize; i++) {
        for (j = 0; j < m_colSize; j++) {
            result(i,j) = this->m_matrix[i][j] + scalar;
        }
    }
    return result;
}

// Scalar Subraction
Matrix Matrix::operator-(double scalar){
    Matrix result(m_rowSize,m_colSize,0.0);
    unsigned i,j;
    for (i = 0; i < m_rowSize; i++) {
        for (j = 0; j < m_colSize; j++) {
            result(i,j) = this->m_matrix[i][j] - scalar;
        }
    }
    return result;
}

// Scalar Multiplication
Matrix Matrix::operator*(double scalar){
    Matrix result(m_rowSize,m_colSize,0.0);
    unsigned i,j;
    for (i = 0; i < m_rowSize; i++) {
        for (j = 0; j < m_colSize; j++) {
            result(i,j) = this->m_matrix[i][j] * scalar;
        }
    }
    return result;
}

// Scalar Division
Matrix Matrix::operator/(double scalar){
    Matrix result(m_rowSize,m_colSize,0.0);
    unsigned i,j;
    for (i = 0; i < m_rowSize; i++) {
        for (j = 0; j < m_colSize; j++) {
            result(i,j) = this->m_matrix[i][j] / scalar;
        }
    }
    return result;
}

// Returns value of given location when asked in the form A(x,y)
double& Matrix::operator()(const unsigned &rowNo, const unsigned & colNo) {
    return this->m_matrix[rowNo][colNo];
}

//___________GETTERS__________
unsigned Matrix::getRows() const {
    return this->m_rowSize;
}

unsigned Matrix::getCols() const {
    return this->m_colSize;
}

// Take any given matrices transpose and returns another matrix
Matrix Matrix::transpose() const {
    Matrix Transpose(m_colSize, m_rowSize, 0.0);
    for (unsigned i = 0; i < m_colSize; i++)
    {
        for (unsigned j = 0; j < m_rowSize; j++) {
            Transpose(i,j) = this->m_matrix[j][i];
        }
    }
    return Transpose;
}


// Matrix Matrix::matdot(const Matrix& m1, const Matrix& m2) const {}


// Prints the matrix beautifully
void Matrix::printMatrix() const {
    cout << "Matrix: " << endl;
    for (unsigned i = 0; i < m_rowSize; i++) {
        for (unsigned j = 0; j < m_colSize; j++) {
            cout << "[" << m_matrix[i][j] << "] ";
        }
        cout << endl;
    }
}
