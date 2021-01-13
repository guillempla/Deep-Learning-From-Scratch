#include "loss.hh"

//___________CONSTRUCTORS__________
Loss::Loss() {}

//___________SETTERS__________


//___________GETTERS__________
double Loss::mean_square(Matrix& y_true, Matrix& y_pred) {
    Matrix loss = (y_true-y_pred).pow2Matrix();
    Matrix aux = (loss.sum(1))/y_pred.getCols();
    return aux.sum()/aux.getRows();
}

Matrix Loss::mean_square_prime(Matrix& y_true, Matrix& y_pred) {
    return (y_true - y_pred)*2;
}

double Loss::cross_entropy(Matrix& y_true, Matrix& y_pred) {
    Matrix y_pred_clip = y_pred.clip(EPSILON, 1-EPSILON);
    Matrix y_pred_log = y_pred_clip.logMatrix();
    Matrix loss =  y_true.mulElementWise(y_pred_log);
    return - (loss.sum(1)).sum();
}

Matrix Loss::cross_entropy_prime(Matrix& y_true, Matrix& y_pred) {
    Matrix y_pred_inverse = y_pred.inverse();
    return y_true.mulElementWise(y_pred_inverse);
}
