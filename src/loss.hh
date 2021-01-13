#ifndef LOSS_HH
#define LOSS_HH

#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.hh"

using namespace std;

#define EPSILON 1e-14

class Loss {
    private:

    public:
        //___________CONSTRUCTORS__________
        Loss();

        //___________SETTERS__________


        //___________GETTERS__________

        /*
        y_true: vector of desired values
        y_pred: vector of values given by the model
        */
        static double mean_square(Matrix& y_true, Matrix& y_pred);
        static double cross_entropy(Matrix& y_true, Matrix& y_pred);

        static Matrix mean_square_prime(Matrix& y_true, Matrix& y_pred);
        static Matrix cross_entropy_prime(Matrix& y_true, Matrix& y_pred);

};

#endif
