#ifndef LOSS_FUNCTIONS_HH
#define LOSS_FUNCTIONS_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Loss_functions {
    private:

    public:
        //___________CONSTRUCTORS__________
        Loss_functions();

        //___________SETTERS__________


        //___________GETTERS__________

        /*
        y_true: vector of desired values
        y_pred: vector of values given by the model (w·x)
        */
        float mean_squared_error(const vector<float>& y_true, const vector<float>& y_pred) const;

        /*
        y_true: vector of desired values
        y_pred: vector of values given by the model (w·x)
        x:      vector of values of previous layers
        */
        float mean_squared_error_prime(const vector<float>& y_true, const vector<float>& y_pred, const vector<float>& x) const;

}

#endif
