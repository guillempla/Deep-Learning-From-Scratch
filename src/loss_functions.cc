#include "loss_functions.hh"

//___________CONSTRUCTORS__________
Loss_functions::Loss_functions() {}

//___________SETTERS__________


//___________GETTERS__________
float Loss_functions::mean_squared_error(const vector<float>& y_true, const vector<float>& y_pred) const {
    float sum = 0.0;
    float p = y_pred.size();
    for (int k = 0; k < p; k++)
        sum += pow(y_pred[i]-y_true[i], 2);
    return 0.5*sum;
}

flot Loss_functions::mean_squared_error_prime(const vector<float>& y_true, const vector<float>& y_pred, const vector<float>& x) const {
    float sum = 0.0;
    float p = y_pred.size();
    for (int k = 0; k < p; k++)
        sum += (y_pred[i]-y_true[i])*x[i];
    return sum;
}
