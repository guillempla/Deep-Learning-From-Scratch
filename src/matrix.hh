#ifndef MATRIX_HH
#define MATRIX_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Matrix {
    private:
        unsigned m, n;
        vector<vector<double> > m_matrix;

    public:
        //___________CONSTRUCTORS__________
        Matrix(unsigned m, unsigned n);
        Matrix(unsigned m, unsigned n, double initial);

        //___________SETTERS__________


        //___________GETTERS__________
        int get_rows() const;
        int get_cols() const;

        Matrix matsum(const Matrix& m1, const Matrix& m2) const;
        Matrix matsub(const Matrix& m1, const Matrix& m2) const;
        Matrix matdiv(const Matrix& m1, const Matrix& m2) const;
        Matrix matmul(const Matrix& m1, const Matrix& m2) const;
        Matrix matdot(const Matrix& m1, const Matrix& m2) const;

        void print_matrix() const;



};

#endif
