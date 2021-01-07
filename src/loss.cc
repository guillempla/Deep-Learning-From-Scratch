#include "loss.hh"

//___________CONSTRUCTORS__________
Loss::Loss() {}

//___________SETTERS__________


//___________GETTERS__________
double Loss::mean_squared_error(const vector<double>& y_true, const vector<double>& y_pred) {
    double sum = 0.0;
    double p = y_pred.size();
    for (int k = 0; k < p; k++)
        sum += pow(y_pred[k]-y_true[k], 2);
    return 0.5*sum;
}

double Loss::cross_entropy(const vector<double>& y_true, const vector<double>& y_pred) {
    double sum = 0.0;
    for (int k = 0; k < y_pred.size(); k++) {
        sum += y_true[k]*log(y_pred[k])+(1-y_true[k])*log(1-y_pred[k]);
    }
    return sum/y_pred.size();
}

double Loss::mean_squared_error_prime(const vector<double>& y_true, const vector<double>& y_pred, const vector<double>& x) {
    double sum = 0.0;
    double p = y_pred.size();
    for (int k = 0; k < p; k++)
        sum += (y_pred[k]-y_true[k])*x[k];
    return sum;
}

vector<double> Loss::cross_entropy_prime(const vector<double>& y_true, const vector<double>& y_pred) {
    vector<double> dAL(y_pred.size());
    for (int k = 0; k < y_pred.size(); k++) {
        double d1 = y_true[k]/y_pred[k];
        double d2 = (1-y_true[k])/(1-y_pred[k]);
        dAL[k] = - (d1-d2);
    }
    return dAL;
}
