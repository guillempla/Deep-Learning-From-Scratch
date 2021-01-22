#include <utility>

#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(string type, unsigned num_examples, unsigned layer_size, unsigned prev_size) {
    this->type = move(type);

    // unsigned seed = 1;

    this->b = Matrix(1, layer_size, 0.0);
    this->W = Matrix(layer_size, prev_size, true);

    // cout << "        Layer::Finished b,W" << endl;

    this->db = Matrix(1, layer_size, 0.0);
    this->dW = Matrix(layer_size, prev_size, 0.0);

    // cout << "        Layer::Finished db,dW" << endl;

    this->Z = Matrix(layer_size, num_examples);
    this->A = Matrix(layer_size, num_examples);

    // cout << "        Layer::Finished Z,A" << endl;

    this->dZ = Matrix(layer_size, num_examples);
    this->dA = Matrix(layer_size, num_examples);

    // cout << "        Layer::Finished dZ,dA" << endl;
}

void Layer::set_activation_gradient(Matrix& dA) {
    this->dA = dA;
}

Matrix Layer::predict(Matrix& A_prev) {
    // cout << "    Layer::predict" << endl;
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
    // cout << "    Layer::feed_forward" << endl;
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

Matrix Layer::back_propagate(Matrix& A_prev) {
    // cout << "    Layer::Initialized back_propagate" << endl;

    activation_backward();

    double m = A_prev.getCols();
    Matrix A_prevT = A_prev.transpose();
    dW = (dZ*A_prevT)/m;
    db = (dZ.sum(1)/m).transpose();
    Matrix dA_prev = W.transpose()*dZ;
    return dA_prev;
}

void Layer::update_parameters(double learning_rate) {
    // cout << "    Layer::update_parameters" << endl;
    Matrix Waux = dW*learning_rate;
    Matrix baux = db*learning_rate;
    W = W + Waux;
    b = b + baux;
}


Matrix* Layer::get_activation() {
    return &(this->A);
}

//___________PRIVATE__________
void Layer::activation_backward() {
    if (type == "sigmoid") {
        // cout << "    Layer::Type Sigmoid" << endl;
        Matrix g_prime = A.sigmoid_prime();
        dZ = dA.mulElementWise(g_prime);
    }
    else if (type == "relu"){
        // cout << "    Layer::Type ReLu" << endl;
        Matrix g_prime = Z.relu_prime();
        dZ = dA.mulElementWise(g_prime);
    }
    else if (type == "softmax") {
        // cout << "    Layer::Type Softmax" << endl;
        Matrix g_prime = Z.softmax_prime();
        dZ = dA.mulElementWise(g_prime);
    }
    else
        throw invalid_argument("ERROR back_propagate: Wrong layer type!");
}
