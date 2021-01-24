#include "loss.hh"

//___________CONSTRUCTORS__________
Loss::Loss() = default;

//___________SETTERS__________


//___________GETTERS__________
double Loss::mean_square(Matrix& y_true, Matrix& y_pred) {
    Matrix loss = (y_true - y_pred).square();
    Matrix aux = (loss.sum(1))/y_pred.getCols();
    return aux.sum()/aux.getRows();
}

Matrix Loss::mean_square_prime(Matrix& y_true, Matrix& y_pred) {
    return (y_true - y_pred)*2;
}

double Loss::cross_entropy(Matrix& y_true, Matrix& y_pred) {
    Matrix y_pred_clip = y_pred.clip(EPSILON, 1-EPSILON);
    Matrix y_pred_log = y_pred_clip.logMatrix();
    Matrix loss = y_true.mulElementWise(y_pred_log);
    return - (loss.sum(0)).sum()/(y_true.getCols()*y_true.getRows());
}

Matrix Loss::cross_entropy_prime(Matrix& y_true, Matrix& y_pred) {
    return y_pred - y_true;
}

double Loss::binary_cross_entropy(Matrix& y_true, Matrix& y_pred) {
    Matrix y_true_comp = y_true*-1.0 + 1.0;
    Matrix y_pred_comp = y_pred*-1.0 + 1.0;

    Matrix y_pred_clip = y_pred.clip(EPSILON, 1-EPSILON);
    Matrix y_pred_comp_clip = y_pred_comp.clip(EPSILON, 1-EPSILON);

    Matrix y_pred_log = y_pred_clip.logMatrix();
    Matrix y_pred_comp_log = y_pred_comp_clip.logMatrix();

    Matrix sum1 = y_true.mulElementWise(y_pred_log);
    Matrix sum2 = y_true_comp.mulElementWise(y_pred_comp_log);
    Matrix loss = sum1 + sum2;
    return - (loss.sum(0)).sum()/y_true.getCols();
}

Matrix Loss::binary_cross_entropy_prime(Matrix& y_true, Matrix& y_pred) {
    Matrix y_pred_comp = y_pred*-1.0 + 1.0;
    Matrix denom = (y_pred.mulElementWise(y_pred_comp));
    Matrix denom_clip = denom.clip(EPSILON, 1-EPSILON);
    return (y_true - y_pred) / denom_clip;
}
