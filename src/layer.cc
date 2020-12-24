#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(int type, int layer_size, int next_size) {
    this->type;
    this->layer_size;
    this->next_size;

    this->initialize_bias();
    this->initialize_weights();
}

//___________SETTERS__________
void Layer::set_type(int type) {
    this->type = type;
}

void Layer::set_bias(const Matrix& bias) {
    this->bias = bias;
}

void Layer::set_weights(const Matrix& weights) {
    this->weights = weights;
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


//___________PRIVATE__________
void Layer::initialize_bias() {

}

void Layer::initialize_weights() {

}
