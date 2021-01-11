#ifndef MATRIX_HH
#define MATRIX_HH

#include <iostream>
#include <vector>
#include <omp.h>
#include <math.h>
#include <stdlib.h> // Random numbers
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
        Matrix(unsigned m, unsigned n, bool randn, unsigned seed);
        Matrix(unsigned m, unsigned n, double initial);

        //___________OPERATIONS__________
        // Matrix operations
        Matrix operator+(Matrix &);
        Matrix operator-(Matrix &);
        Matrix operator*(Matrix &);
        Matrix operator/(Matrix &);

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

        // Given a Matrix returns its transpose
        Matrix transpose() const;

        // Given two Matrices returns an elementwise multiplication of both
        Matrix mulElementWise(Matrix& m) const;

        // Given two unidimensional Matrices returns dot product
        double dot(Matrix& m) const;

        // Given one unidimensional Matrix returns the sum of its elements
        double sum() const;

        // Given a Matrix returns an unidimensional Matrix containing the sums of axis
        Matrix sum(int axis) const;

        // Given one unidimensional Matrix applies exp() to all elements
        Matrix expMatrix() const;

        // Given a Matrix returns a Matrix with all its elements multiplied by itself
        Matrix pow2Matrix() const;

        //___________ACTIVATION__________
        Matrix sigmoid();
        Matrix relu();
        Matrix softmax();

        // Matrix sigmoid_prime();
        Matrix relu_prime();

        //___________PRINT__________
        // Prints the matrix beautifully
        void printMatrix() const;



};

#endif
