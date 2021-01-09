#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(bool type, unsigned num_examples, unsigned layer_size, unsigned prev_size) {
    this->type;
    this->layer_size;
    this->prev_size;
    this->num_examples;

    this->bias = Matrix(1, layer_size, 0.0);
    this->weights = Matrix(layer_size, prev_size, 0.01);

    this->bias_prime = Matrix(1, layer_size, 0.0);
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

Matrix* Layer::feed_forward(Matrix& A_prev) {
    this->potential = this->weights*A_prev + this->bias;
    if (type)
        this->activation = this->potential.sigmoid();
    else
        this->activation = this->potential.relu();
    return &(this->activation);
}

Matrix* Layer::back_propagate(Matrix& dA_prev) {
    // A_prev, W, b = cache
    // m = A_prev.shape[1]
    //
    // dW = np.dot(dZ, A_prev.T)/m
    // db = np.sum(dZ, axis=1, keepdims=True)/m
    // dA_prev = np.dot(W.T, dZ)
    // this->activation_prime = ;
    if (type)
        this->potential_prime = this->potential.sigmoid_prime();
    else
        this->potential_prime = this->potential.relu_prime();
    return &(this->activation_prime);
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

}
