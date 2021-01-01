#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(int type, unsigned layer_size, unsigned next_size) {
    this->type;
    this->layer_size;
    this->next_size;

    this->bias = Matrix(this->layer_size, (unsigned)(1), 0.01);
    this->weights = Matrix(this->layer_size, this->next_size, 0.01);

    this->Z = Matrix(this->layer_size,  (unsigned)(1));
    this->activation = Matrix(this->layer_size,  (unsigned)(1));
}

//___________SETTERS__________
void Layer::set_type(int type) {
    this->type = type;
}

void Layer::set_layer_size(unsigned layer_size) {
    this->layer_size = layer_size;
}

void Layer::set_next_size(unsigned next_size) {
    this->next_size = next_size;
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

void Layer::linear_activation_forward(Matrix& A_prev) {
    this->Z = this->weights*A_prev + this->bias;

    Activation_functions a(1);

    for (int i = 0; i < Z.getRows(); i++) {
        if (type == 0 || type == 1)
            this->activation(i,1) = a.relu(this->Z(i,1));
        else
            this->activation(i,1) = a.sigmoid(this->Z(i,1));
    }
}



//___________GETTERS__________
int Layer::get_type() const {
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

Matrix Layer::get_activation() const {
    return this->activation;
}
