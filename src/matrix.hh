#ifndef MATRIX_HH
#define MATRIX_HH

#include <iostream>
#include <vector>
#include <omp.h>
#include <math.h>
#include "activation.hh"

using namespace std;

class Matrix {
    private:
        unsigned m_rowSize;
        unsigned m_colSize;
        vector<vector<double> > m_matrix;

    public:
        //___________CONSTRUCTORS__________
        Matrix();
        Matrix(unsigned m, unsigned n);
        Matrix(unsigned m, unsigned n, double initial);

        //___________SETTERS__________


        //___________OPERATIONS__________
        // Matrix operations
        Matrix operator+(Matrix &);
        Matrix operator-(Matrix &);
        Matrix operator*(Matrix &);

        // Scalar Operations
        Matrix operator+(double);
        Matrix operator-(double);
        Matrix operator*(double);
        Matrix operator/(double);

        // Returns value of given location when asked in the form A(x,y)
        double& operator()(const unsigned &);
        // Returns value of given location when asked in the form A(x)
        double& operator()(const unsigned &, const unsigned &);

        //___________GETTERS__________
        unsigned getRows() const;
        unsigned getCols() const;

        Matrix transpose() const;

        // Given two unidimensional Matrices returns dot product
        double dot(Matrix& m) const;

        //___________ACTIVATION__________
        Matrix sigmoid() const;
        Matrix relu() const;

        Matrix sigmoid_prime() const;
        Matrix relu_prime() const;

        //___________LOSS__________
        Matrix mean_squared_error() const;
        Matrix cross_entropy() const;

        Matrix mean_squared_error_prime() const;
        Matrix cross_entropy_prime() const;

        //___________PRINT__________
        // Prints the matrix beautifully
        void printMatrix() const;



};

#endif
