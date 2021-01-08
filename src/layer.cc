#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(bool type, unsigned num_examples, unsigned layer_size, unsigned prev_size) {
    this->type;
    this->layer_size;
    this->prev_size;
    this->num_examples;

    this->bias = Matrix(layer_size, (unsigned)(1), 0.01);
    this->weights = Matrix(layer_size, prev_size, 0.01);

    this->bias_prime = Matrix(layer_size, (unsigned)(1), 0.0);
    this->weights_prime = Matrix(layer_size, prev_size, 0.0);

    this->potential = Matrix(layer_size,  num_examples);
    this->activation = Matrix(layer_size,  num_examples);

    this->potential_prime = Matrix(layer_size,  num_examples);
    this->activation_prime = Matrix(layer_size,  num_examples);
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

void Layer::set_bias(const Matrix& bias) {
    this->bias = bias;
}

void Layer::set_weights(const Matrix& weights) {
    this->weights = weights;
}

void Layer::set_weights_prime(const Matrix& weights_prime) {
    this->weights_prime = weights_prime;
}

void Layer::feed_forward(Matrix& A_prev) {
    this->forward(A_prev);
}

Matrix Layer::back_propagate(Matrix& dA_prev) {

}



//___________GETTERS__________
bool Layer::get_type() const {
    return this->type;
}

Matrix Layer::get_bias() const {
    return this->bias;
}

Matrix Layer::get_weights() const {
    return this->weights;
}

Matrix Layer::get_weights_prime() const {
    return this->weights_prime;
}

Matrix* Layer::get_activation() {
    return &(this->activation);
}


//___________PRIVATE__________
void Layer::forward(Matrix& A_prev) {
    for (int i = 0; i < this->weights.getRows(); i++) {
        for (int j = 0; j < this->weights.getCols(); j++) {
            this->potential(i,0) += this->weights(i,j)*A_prev(i,j) + this->bias(i,0);
        }
    }
}
