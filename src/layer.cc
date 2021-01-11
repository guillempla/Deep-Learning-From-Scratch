#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(bool type, unsigned num_examples, unsigned layer_size, unsigned prev_size) {
    this->type;
    this->layer_size;
    this->prev_size;
    this->num_examples;

    unsigned seed = 1;

    this->b = Matrix(1, layer_size, 0.0);
    this->W = Matrix(layer_size, prev_size, false, seed);

    this->db = Matrix(1, layer_size, 0.0);
    this->dW = Matrix(layer_size, prev_size, 0.0);

    this->Z = Matrix(layer_size, num_examples);
    this->A = Matrix(layer_size, num_examples);

    this->dZ = Matrix(layer_size, num_examples);
    this->dA = Matrix(layer_size, num_examples);
}

//___________SETTERS__________
void Layer::set_type(bool type) {
    this->type = type;
}

void Layer::set_layer_size(unsigned layer_size) {
    this->layer_size = layer_size;
}

void Layer::set_prev_size(unsigned prev_size) {
    this->prev_size = prev_size;
}

void Layer::set_bias(const Matrix& b) {
    this->b = b;
}

void Layer::set_weights(const Matrix& W) {
    this->W = W;
}

void Layer::set_weights_gradient(const Matrix& dW) {
    this->dW = dW;
}

void Layer::set_activation_gradient(const Matrix& dA) {
    this->dA = dA;
}

Matrix* Layer::feed_forward(Matrix& A_prev) {
    Z = W*A_prev + b;
    if (type)
        A = Z.sigmoid();
    else
        A = Z.relu();
    return &A;
}

Matrix* Layer::back_propagate(Matrix& dA) {
    // if (type)
    //     dZ = dA.sigmoid_prime();
    // else
    //     dZ = dA.relu_prime();

    unsigned m = A_prev.getCols();
    auto A_prevT = A_prev.transpose();
    dW = (dZ*A_prevT)/m;
    db = (dZ.sum(1)/m).transpose();
    // dA_prev = W.transpose()*dZ;
    // return &dA_prev;
}



//___________GETTERS__________
bool Layer::get_type() const {
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
