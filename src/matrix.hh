#ifndef MATRIX_HH
#define MATRIX_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Matrix {
    private:
        unsigned m_rowSize;
        unsigned m_colSize;
        vector<vector<double> > m_matrix;

    public:
        //___________CONSTRUCTORS__________
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
        double& operator()(const unsigned &, const unsigned &);

        //___________GETTERS__________
        unsigned getRows() const;
        unsigned getCols() const;

        Matrix transpose();

        void printMatrix() const;

        // Matrix matdot(const Matrix& m1, const Matrix& m2) const;




};

#endif
