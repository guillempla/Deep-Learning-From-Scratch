#ifndef MATRIX_HH
#define MATRIX_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Matrix {
    private:
        int m, n;
        vector<vector<double>> m_matrix;

    public:
        //___________CONSTRUCTORS__________
        Matrix(int m, int n);

        //___________SETTERS__________


        //___________GETTERS__________
        Matrix matsum(const Matrix& m1, const Matrix& m2) const;
        Matrix matsub(const Matrix& m1, const Matrix& m2) const;
        Matrix matdiv(const Matrix& m1, const Matrix& m2) const;
        Matrix matmul(const Matrix& m1, const Matrix& m2) const;
        Matrix matdot(const Matrix& m1, const Matrix& m2) const;

        static void print_matrix(const Matrix& m);



};

#endif
