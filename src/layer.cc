#include <utility>

#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(string type, unsigned num_examples, unsigned layer_size, unsigned prev_size) {
    this->type = move(type);

    this->b = Matrix(1, layer_size, 0.0);
    this->W = Matrix(layer_size, prev_size, true);

    this->db = Matrix(1, layer_size, 0.0);
    this->dW = Matrix(layer_size, prev_size, 0.0);

    this->Z = Matrix(layer_size, num_examples);
    this->A = Matrix(layer_size, num_examples);

    this->dZ = Matrix(layer_size, num_examples);
    this->dA = Matrix(layer_size, num_examples);
}

void Layer::set_activation_gradient(Matrix& dA) {
    this->dA = dA;
}

Matrix Layer::predict(Matrix& A_prev) {
    Matrix Zaux = W*A_prev + b;
    if (type == "sigmoid")
        return Zaux.sigmoid();
    else if (type == "relu")
        return Zaux.relu();
    else if (type == "softmax")
        return Zaux.softmax();
    else
    throw invalid_argument("ERROR Predict: Wrong layer type!");
}

Matrix* Layer::feed_forward(Matrix& A_prev) {
    Z = W*A_prev + b;
    if (type == "sigmoid")
        A = Z.sigmoid();
    else if (type == "relu")
        A = Z.relu();
    else if (type == "softmax")
        A = Z.softmax();
    else
        throw invalid_argument("ERROR Predict: Wrong layer type!");
    return &A;
}

Matrix Layer::back_propagate(Matrix& A_prev, double lambd) {
    activation_backward();

    double m = A_prev.getCols();
    Matrix A_prevT = A_prev.transpose();
    Matrix W_l2 = W*(lambd/m);
    dW = (dZ*A_prevT)/m + W_l2;
    db = (dZ.sum(1)/m).transpose();
    Matrix dA_prev = W.transpose()*dZ;
    return dA_prev;
}

void Layer::update_parameters(double learning_rate) {
    Matrix Waux = dW*learning_rate;
    Matrix baux = db*learning_rate;
    W = W + Waux;
    b = b + baux;
}

Matrix *Layer::get_weights() {
    return &W;
}

Matrix* Layer::get_activation() {
    return &A;
}

//___________PRIVATE__________
void Layer::activation_backward() {
    if (type == "sigmoid") {
        Matrix g_prime = A.sigmoid_prime();
        dZ = dA.mulElementWise(g_prime);
    }
    else if (type == "relu"){
        Matrix g_prime = Z.relu_prime();
        dZ = dA.mulElementWise(g_prime);
    }
    else if (type == "softmax") {
        // This is not mathematically correct but its simplier
        dZ = dA;
    }
    else
        throw invalid_argument("ERROR back_propagate: Wrong layer type!");
}
