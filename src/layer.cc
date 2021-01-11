#include "layer.hh"

//___________CONSTRUCTORS__________
Layer::Layer(string type, unsigned num_examples, unsigned layer_size, unsigned prev_size) {
    cout << "        Layer::Initializing layer" << endl;
    this->type;
    cout << "        Layer::Initialized type: " << type << endl;
    this->layer_size;
    this->prev_size;
    this->num_examples;

    cout << "        Layer::Finished init atributes" << endl;

    unsigned seed = 1;

    this->b = Matrix(1, layer_size, 0.0);
    this->W = Matrix(layer_size, prev_size, false, seed);

    cout << "        Layer::Finished b,W" << endl;

    this->db = Matrix(1, layer_size, 0.0);
    this->dW = Matrix(layer_size, prev_size, 0.0);

    cout << "        Layer::Finished db,dW" << endl;

    this->Z = Matrix(layer_size, num_examples);
    this->A = Matrix(layer_size, num_examples);

    cout << "        Layer::Finished Z,A" << endl;

    this->dZ = Matrix(layer_size, num_examples);
    this->dA = Matrix(layer_size, num_examples);

    cout << "        Layer::Finished dZ,dA" << endl;
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

Matrix* Layer::feed_forward(Matrix& A_prev) {
    cout << "    Layer::feed_forward" << endl;
    cout << "    Layer::Type: " << type << endl;
    cout << "    Layer::A_prev dimensions(" << A_prev.getRows() << "," << A_prev.getCols() << ")" << endl;
    cout << "    Layer::W dimensions(" << W.getRows() << "," << W.getCols() << ")" << endl;
    cout << "    Layer::b dimensions(" << b.getRows() << "," << b.getCols() << ")" << endl;
    Z = W*A_prev + b;
    cout << "    Layer::Z calculated" << endl;
    cout << "    Layer::Z dimensions(" << Z.getRows() << "," << Z.getCols() << ")" << endl;
    if (this->type == "output")
        A = Z.sigmoid();
    else
        A = Z.relu();
    cout << "    Layer::A calculated" << endl;
    return &A;
}

Matrix Layer::back_propagate(Matrix& A_prev) {
    if (type == "output") {
        Matrix g_prime = Z.sigmoid_prime();
        dZ = dA*g_prime;
    }
    else {
        Matrix g_prime = Z.relu_prime();
        dZ = dA*g_prime;
    }

    unsigned m = A_prev.getCols();
    Matrix A_prevT = A_prev.transpose();
    dW = (dZ*A_prevT)/m;
    db = (dZ.sum(1)/m).transpose();
    Matrix dA_prev = W.transpose()*dZ;
    return dA_prev;
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
