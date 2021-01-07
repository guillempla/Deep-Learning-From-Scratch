#include "matrix.hh"
#include <stdexcept>

using namespace std;

//___________CONSTRUCTORS__________
Matrix::Matrix() {}

Matrix::Matrix(unsigned m, unsigned n) {
    m_rowSize = m;
    m_colSize = n;
    this->m_matrix.resize(m);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m; i++)
        m_matrix[i].resize(n, 0);
}

Matrix::Matrix(unsigned m, unsigned n, double initial) {
    m_rowSize = m;
    m_colSize = n;
    this->m_matrix.resize(m);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m; i++)
        m_matrix[i].resize(n, initial);
}


//___________SETTERS__________



//___________OPERATIONS__________
// Addition of Two Matrices
Matrix Matrix::operator+(Matrix& B){
    Matrix sum(m_colSize, m_rowSize, 0.0);
    unsigned i,j;
    #pragma omp parallel for num_threads(16)
    for (i = 0; i < m_rowSize; i++) {
        for (j = 0; j < m_colSize; j++) {
            sum(i,j) = this->m_matrix[i][j] + B(i,j);
        }
    }
    return sum;
}

// Subtraction of Two Matrices
Matrix Matrix::operator-(Matrix& B){
    Matrix diff(m_colSize, m_rowSize, 0.0);
    unsigned i,j;
    #pragma omp parallel for num_threads(16)
    for (i = 0; i < m_rowSize; i++) {
        for (j = 0; j < m_colSize; j++) {
            diff(i,j) = this->m_matrix[i][j] - B(i,j);
        }
    }

    return diff;
}

// Multiplication of Two Matrices
Matrix Matrix::operator*(Matrix& B){
    Matrix multip(m_rowSize, B.getCols(),0.0);
    if (m_colSize == B.getRows()) {
        unsigned i,j,k;
        double temp = 0.0;
        #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
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

// Returns value of given location when asked in the form A(x)
double& Matrix::operator()(const unsigned & colNo) {
    if (this->m_rowSize == 1 && this->m_colSize > colNo)
        return this->m_matrix[0][colNo];
    else
        throw std::invalid_argument("ERROR: Matrix m(x) must be unidimensional");
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
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_colSize; i++) {
        for (unsigned j = 0; j < m_rowSize; j++)
            Transpose(i,j) = this->m_matrix[j][i];
    }
    return Transpose;
}

double Matrix::dot(Matrix& m) const {
    double res = 0.0;
    if (this->m_rowSize == m.getRows() && this->m_colSize == 1 && m.getCols() == 1) {
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m.getRows(); i++)
            res += this->m_matrix[i][0]*m(i,0);
    }
    else if (this->m_rowSize != m.getRows() && this->m_colSize == 1 && m.getCols() != 1) {
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m.getRows(); i++)
            res += this->m_matrix[i][0]*m(0,i);
    }
    else if (this->m_rowSize != m.getRows() && this->m_colSize != 1 && m.getCols() == 1) {
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m.getRows(); i++)
            res += this->m_matrix[0][i]*m(i,0);
    }
    else if (this->m_colSize == m.getCols() && this->m_rowSize == 1 && m.getRows() == 1) {
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m.getRows(); i++)
            res += this->m_matrix[0][i]*m(0,i);
    }
    else
        throw std::invalid_argument("ERROR: Dot product vectors must be unidimensional");
    return res;
}


//___________ACTIVATION__________
void Matrix::sigmoid() {
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            this->m_matrix[i][j] = Activation::sigmoid(m_matrix[i][j]);
}

void Matrix::relu() {
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            this->m_matrix[i][j] = Activation::relu(m_matrix[i][j]);
}


void Matrix::sigmoid_prime() {
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            this->m_matrix[i][j] = Activation::sigmoid_prime(m_matrix[i][j]);
}

void Matrix::relu_prime() {
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            this->m_matrix[i][j] = Activation::relu_prime(m_matrix[i][j]);
}
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
