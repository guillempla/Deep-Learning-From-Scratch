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

Matrix::Matrix(unsigned m, unsigned n, bool randn, unsigned seed) {
    m_rowSize = m;
    m_colSize = n;
    this->m_matrix.resize(m);

    srand(seed);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m; i++) {
        m_matrix[i].resize(n, 0);
        for (unsigned j = 0; j < n; j++) {
            if (!randn)
                m_matrix[i][j] = (double)(rand()%100)/100;
        }
    }
}

Matrix::Matrix(unsigned m, unsigned n, double initial) {
    m_rowSize = m;
    m_colSize = n;
    this->m_matrix.resize(m);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m; i++)
        m_matrix[i].resize(n, initial);
}



//___________OPERATIONS__________
// Addition of Two Matrices
Matrix Matrix::operator+(Matrix& B){
    Matrix sum(m_rowSize, m_colSize, 0.0);

    if (m_rowSize == B.getRows() && m_colSize == B.getCols()) {
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m_rowSize; i++)
            for (unsigned j = 0; j < m_colSize; j++)
                sum(i,j) = this->m_matrix[i][j] + B(i,j);
    }
    else if (m_rowSize == B.getCols() && B.getRows() == 1) {
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m_rowSize; i++) {
            double b = B(i);
            for (unsigned j = 0; j < m_colSize; j++)
                sum(i,j) = this->m_matrix[i][j] + b;
        }
    }
    else
        throw invalid_argument("ERROR +: Wrong matrix dimension!");
    return sum;
}

// Subtraction of Two Matrices
Matrix Matrix::operator-(Matrix& B){
    Matrix diff(m_rowSize, m_colSize, 0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++) {
        for (unsigned j = 0; j < m_colSize; j++) {
            diff(i,j) = this->m_matrix[i][j] - B(i,j);
        }
    }
    return diff;
}

// Multiplication of Two Matrices
Matrix Matrix::operator*(Matrix& B){
    Matrix multip(m_rowSize, B.getCols(),0.0);
    if (m_colSize == B.getRows()) {
        double temp = 0.0;
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m_rowSize; i++) {
            for (unsigned j = 0; j < B.getCols(); j++) {
                temp = 0.0;
                for (unsigned k = 0; k < m_colSize; k++) {
                    temp += m_matrix[i][k] * B(k,j);
                }
                multip(i,j) = temp;
            }
        }
        return multip;
    }
    else {
        throw invalid_argument("ERROR *: Wrong matrix dimension!");
    }
}

// Division of two matrices
Matrix Matrix::operator/(Matrix& B){
    Matrix result(m_rowSize,m_colSize,0.0);
    if (m_rowSize == B.getRows() && m_colSize == B.getCols()) {
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m_rowSize; i++)
            for (unsigned j = 0; j < m_colSize; j++)
                result(i,j) = this->m_matrix[i][j] / B(i,j);
    }
    else if (m_colSize == B.getCols() && B.getRows() == 1) {
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m_rowSize; i++)
            for (unsigned j = 0; j < m_colSize; j++)
                result(i,j) = this->m_matrix[i][j] / B(j);
    }
    else
        throw invalid_argument("ERROR /: Wrong matrix dimension!");
    return result;
}

// Scalar Addition
Matrix Matrix::operator+(double scalar){
    Matrix result(m_rowSize,m_colSize,0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            result(i,j) = this->m_matrix[i][j] + scalar;
    return result;
}

// Scalar Subraction
Matrix Matrix::operator-(double scalar){
    Matrix result(m_rowSize,m_colSize,0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            result(i,j) = this->m_matrix[i][j] - scalar;
    return result;
}

// Scalar Multiplication
Matrix Matrix::operator*(double scalar){
    Matrix result(m_rowSize,m_colSize,0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            result(i,j) = m_matrix[i][j] * scalar;
    return result;
}

// Scalar Division
Matrix Matrix::operator/(double scalar){
    Matrix result(m_rowSize,m_colSize,0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            result(i,j) = m_matrix[i][j] / scalar;
    return result;
}

// Returns value of given location when asked in the form A(x,y)
double& Matrix::operator()(const unsigned &rowNo, const unsigned & colNo) {
    return this->m_matrix[rowNo][colNo];
}

// Returns value of given location when asked in the form A(x)
double& Matrix::operator()(const unsigned & colNo) {
    if (m_rowSize == 1 && m_colSize > colNo)
        return m_matrix[0][colNo];
    else if (m_colSize == 1 && m_rowSize > colNo)
        return m_matrix[colNo][0];
    else
        throw invalid_argument("ERROR (): Matrix m(x) must be unidimensional");
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
        throw invalid_argument("ERROR DOT: Dot product vectors must be unidimensional");
    return res;
}

double Matrix::sum() const {
    if (m_rowSize == 1) {
        double sum = 0.0;
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m_colSize; i++)
            sum += m_matrix[0][i];
        return sum;
    }
    else
        throw invalid_argument("ERROR SUM: Matrix m must be unidimensional");
}

Matrix Matrix::sum(int axis) const {
    if (axis == 1) {
        Matrix sum(1, m_colSize);
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m_colSize; i++)
            for (unsigned j = 0; j < m_rowSize; j++)
                sum(i) += m_matrix[j][i];
        return sum;
    }
    else {
        Matrix sum(m_rowSize, 1);
        #pragma omp parallel for num_threads(16)
        for (unsigned i = 0; i < m_rowSize; i++)
            for (unsigned j = 0; j < m_colSize; j++)
                sum(i) += m_matrix[i][j];
        return sum;
    }
}

Matrix Matrix::expMatrix() const {
    Matrix res(m_rowSize, m_colSize);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            res(i,j) = exp(m_matrix[i][j]);
    return res;
}


//___________ACTIVATION__________
Matrix Matrix::sigmoid() const {
    Matrix activation(m_rowSize, m_colSize, 0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            activation(i,j) = Activation::sigmoid(m_matrix[i][j]);
    return activation;
}

Matrix Matrix::relu() const {
    Matrix activation(m_rowSize, m_colSize, 0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            activation(i,j) = Activation::relu(m_matrix[i][j]);
    return activation;
}

Matrix Matrix::softmax() const {
    Matrix numerator = this->expMatrix();
    numerator.printMatrix();
    Matrix sum = numerator.sum(1);
    sum.printMatrix();
    return numerator/sum;
}


Matrix Matrix::sigmoid_prime() const {
    Matrix activation(m_rowSize, m_colSize, 0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            activation(i,j) = Activation::sigmoid_prime(m_matrix[i][j]);
    return activation;
}

Matrix Matrix::relu_prime() const {
    Matrix activation(m_rowSize, m_colSize, 0.0);
    #pragma omp parallel for num_threads(16)
    for (unsigned i = 0; i < m_rowSize; i++)
        for (unsigned j = 0; j < m_colSize; j++)
            activation(i,j) = Activation::relu_prime(m_matrix[i][j]);
    return activation;
}


//___________LOSS__________
// Matrix Matrix::mean_squared_error() const {
//     // #pragma omp parallel for num_threads(16)
//     // for (unsigned i = 0; i < m_rowSize; i++)
//     // for (unsigned j = 0; j < m_colSize; j++)
//     // this->m_matrix[i][j] = Loss::mean_squared_error(m_matrix[i][j]);
// }
//
// Matrix Matrix::cross_entropy() const {
//     // #pragma omp parallel for num_threads(16)
//     // for (unsigned i = 0; i < m_rowSize; i++)
//     // for (unsigned j = 0; j < m_colSize; j++)
//     // this->m_matrix[i][j] = Loss::cross_entropy(m_matrix[i][j]);
// }
//
//
// Matrix Matrix::mean_squared_error_prime() const {
//     // #pragma omp parallel for num_threads(16)
//     // for (unsigned i = 0; i < m_rowSize; i++)
//     // for (unsigned j = 0; j < m_colSize; j++)
//     // this->m_matrix[i][j] = Loss::mean_squared_error_prime(m_matrix[i][j]);
// }
//
// Matrix Matrix::cross_entropy_prime() const {
//     // #pragma omp parallel for num_threads(16)
//     // for (unsigned i = 0; i < m_rowSize; i++)
//     // for (unsigned j = 0; j < m_colSize; j++)
//     // this->m_matrix[i][j] = Loss::cross_entropy_prime(m_matrix[i][j]);
// }



//___________PRINT__________
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
