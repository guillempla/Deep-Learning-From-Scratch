#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(string type, unsigned num_examples, unsigned layer_size, unsigned prev_size) {
    this->type = type;
    this->num_examples = num_examples;
    this->layer_size = layer_size;
    this->prev_size = prev_size;

    unsigned seed = 1;

    this->b = Matrix(1, layer_size, 0.0);
    this->W = Matrix(layer_size, prev_size, false, seed);
    this->W = W-0.5;

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

//___________SETTERS__________
void Layer::set_type(string type) {
    this->type = type;
}

void Layer::set_layer_size(unsigned layer_size) {
    this->layer_size = layer_size;
}

void Layer::set_prev_size(unsigned prev_size) {
    this->prev_size = prev_size;
}

void Layer::set_bias(Matrix& b) {
    this->b = b;
}

void Layer::set_weights(Matrix& W) {
    this->W = W;
}

void Layer::set_weights_gradient(Matrix& dW) {
    this->dW = dW;
}

void Layer::set_activation_gradient(Matrix& dA) {
    this->dA = dA;
}

Matrix Layer::predict(Matrix& A_prev) {
    // cout << "    Layer::predict" << endl;
    // cout << "    Layer::Type: " << this->type << endl;
    // cout << "    Layer::A_prev dimensions(" << A_prev.getRows() << "," << A_prev.getCols() << ")" << endl;
    // cout << "    Layer::W dimensions(" << W.getRows() << "," << W.getCols() << ")" << endl;
    // cout << "    Layer::b dimensions(" << b.getRows() << "," << b.getCols() << ")" << endl;
    Matrix Zaux = W*A_prev + b;
    // cout << "    Layer::Zaux dimensions(" << Zaux.getRows() << "," << Zaux.getCols() << ")" << endl;
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
    // cout << "    Layer::Type: " << this->type << endl;
    // cout << "    Layer::A_prev dimensions(" << A_prev.getRows() << "," << A_prev.getCols() << ")" << endl;
    // cout << "    Layer::W dimensions(" << W.getRows() << "," << W.getCols() << ")" << endl;
    // cout << "    Layer::b dimensions(" << b.getRows() << "," << b.getCols() << ")" << endl;
    Z = W*A_prev + b;
    // cout << "    Layer::Z dimensions(" << Z.getRows() << "," << Z.getCols() << ")" << endl;
    if (type == "sigmoid")
        A = Z.sigmoid();
    else if (type == "relu")
        A = Z.relu();
    else if (type == "softmax")
        A = Z.softmax();
    // cout << "    Layer::A dimensions(" << A.getRows() << "," << A.getCols() << ")" << endl;
    else
        throw invalid_argument("ERROR Predict: Wrong layer type!");
    return &A;
}

Matrix Layer::back_propagate(Matrix& A_prev) {
    // cout << "    Layer::Initialized back_propagate" << endl;
    // cout << "    Layer::Layer_size: " << layer_size << " Prev_size: " << prev_size << endl;

    activation_backward();
    // cout << "    Layer::dZ dimensions(" << dZ.getRows() << "," << dZ.getCols() << ")" << endl;

    double m = A_prev.getCols();
    Matrix A_prevT = A_prev.transpose();
    // cout << "    Layer::A_prevT dimensions(" << A_prevT.getRows() << "," << A_prevT.getCols() << ")" << endl;
    dW = (dZ*A_prevT)/m;
    // cout << "    Layer::dW dimensions(" << dW.getRows() << "," << dW.getCols() << ")" << endl;
    db = (dZ.sum(1)/m).transpose();
    // cout << "    Layer::db dimensions(" << db.getRows() << "," << db.getCols() << ")" << endl;
    Matrix dA_prev = W.transpose()*dZ;
    // cout << "    Layer::dA_prev dimensions(" << dA_prev.getRows() << "," << dA_prev.getCols() << ")" << endl;
    return dA_prev;
}

void Layer::update_parameters(double learning_rate) {
    // cout << "    Layer::update_parameters" << endl;
    Matrix Waux = dW*learning_rate;
    Matrix baux = db*learning_rate;
    W = W - Waux;
    b = b - baux;
    // cout << "    Layer::finished updating for layer" << endl;
}



//___________GETTERS__________
string Layer::get_type() const {
    return this->type;
}

Matrix Layer::get_bias() const {
    return this->b;
}

Matrix Layer::get_weights() const {
    return this->W;
}

Matrix Layer::get_weights_gradient() const {
    return this->dW;
}

Matrix* Layer::get_activation() {
    return &(this->A);
}

Matrix* Layer::get_activation_gradient() {
    return &(this->dA);
}

//___________PRIVATE__________
void Layer::activation_backward() {
    if (type == "sigmoid") {
        // cout << "    Layer::Type Sigmoid" << endl;
        Matrix aux = (A*-1.0)+1.0;
        Matrix g_prime = A.mulElementWise(aux);
        dZ = dA.mulElementWise(g_prime);
    }
    else if (type == "relu"){
        // cout << "    Layer::Type ReLu" << endl;
        Matrix g_prime = Z.relu_prime();
        // cout << "    Layer::g_prime dimensions(" << g_prime.getRows() << "," << g_prime.getCols() << ")" << endl;
        // cout << "    Layer::dA dimensions(" << dA.getRows() << "," << dA.getCols() << ")" << endl;
        dZ = dA.mulElementWise(g_prime);
    }
    else if (type == "softmax") {
        // cout << "    Layer::Type Softmax" << endl;
        Matrix g_prime = Z.softmax_prime();
        // cout << "    Layer::g_prime dimensions(" << g_prime.getRows() << "," << g_prime.getCols() << ")" << endl;
        // cout << "    Layer::dA dimensions(" << dA.getRows() << "," << dA.getCols() << ")" << endl;
        dZ = dA.mulElementWise(g_prime);
    }
    else
        throw invalid_argument("ERROR back_propagate: Wrong layer type!");
}
