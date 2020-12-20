#include "loss_functions.hh"

//___________CONSTRUCTORS__________
Loss_functions::Loss_functions() {}

//___________SETTERS__________


//___________GETTERS__________
double Loss_functions::mean_squared_error(const vector<double>& y_true, const vector<double>& y_pred) const {
    double sum = 0.0;
    double p = y_pred.size();
    for (int k = 0; k < p; k++)
        sum += pow(y_pred[i]-y_true[i], 2);
    return 0.5*sum;
}

double Loss_functions::mean_squared_error_prime(const vector<double>& y_true, const vector<double>& y_pred, const vector<double>& x) const {
    double sum = 0.0;
    double p = y_pred.size();
    for (int k = 0; k < p; k++)
        sum += (y_pred[i]-y_true[i])*x[i];
    return sum;
}
