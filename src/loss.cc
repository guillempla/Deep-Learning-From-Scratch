#include "loss.hh"

//___________CONSTRUCTORS__________
Loss::Loss() {}

//___________SETTERS__________


//___________GETTERS__________
double Loss::mean_square(Matrix& y_true, Matrix& y_pred) {
    Matrix aux = (((y_true-y_pred).pow2Matrix()).sum(1))/(y_pred.getCols());
    return aux.sum()/aux.getRows();
}

Matrix Loss::mean_square_prime(Matrix& y_true, Matrix& y_pred) {
    return (y_true - y_pred)*2;
}
//
// double Loss::cross_entropy(Matrix& y_true, Matrix& y_pred) {
//     // double sum = 0.0;
//     // for (int k = 0; k < y_pred.size(); k++) {
//     //     sum += y_true[k]*log(y_pred[k])+(1-y_true[k])*log(1-y_pred[k]);
//     // }
//     // return sum/y_pred.size();
// }
//
// Matrix Loss::cross_entropy_prime(Matrix& y_true, Matrix& y_pred) {
//     // Matrix dAL(y_pred.size());
//     // for (int k = 0; k < y_pred.size(); k++) {
//     //     double d1 = y_true[k]/y_pred[k];
//     //     double d2 = (1-y_true[k])/(1-y_pred[k]);
//     //     dAL[k] = - (d1-d2);
//     // }
//     // return dAL;
// }
