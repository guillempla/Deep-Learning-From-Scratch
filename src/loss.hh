#ifndef LOSS_HH
#define LOSS_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Loss {
    private:

    public:
        //___________CONSTRUCTORS__________
        Loss();

        //___________SETTERS__________


        //___________GETTERS__________

        /*
        y_true: vector of desired values
        y_pred: vector of values given by the model (w·x)
        */
        static double mean_squared_error(const vector<double>& y_true, const vector<double>& y_pred) const;
        static double cross_entropy(const vector<double>& y_true, const vector<double>& y_pred) const;

        /*
        y_true: vector of desired values
        y_pred: vector of values given by the model (w·x)
        x:      vector of values of previous layers
        */
        static double mean_squared_error_prime(const vector<double>& y_true, const vector<double>& y_pred, const vector<double>& x) const;
        static vector<double> cross_entropy_prime(const vector<double>& y_true, const vector<double>& y_pred) const;

};

#endif
