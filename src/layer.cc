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
    if (this->type == "output")
        return Zaux.sigmoid();
    else
        return Zaux.relu();
}

Matrix* Layer::feed_forward(Matrix& A_prev) {
    // cout << "    Layer::feed_forward" << endl;
    // cout << "    Layer::Type: " << this->type << endl;
    // cout << "    Layer::A_prev dimensions(" << A_prev.getRows() << "," << A_prev.getCols() << ")" << endl;
    // cout << "    Layer::W dimensions(" << W.getRows() << "," << W.getCols() << ")" << endl;
    // cout << "    Layer::b dimensions(" << b.getRows() << "," << b.getCols() << ")" << endl;
    Z = W*A_prev + b;
    // cout << "    Layer::Z dimensions(" << Z.getRows() << "," << Z.getCols() << ")" << endl;
    if (this->type == "output")
        A = Z.sigmoid();
    else
        A = Z.relu();
    // cout << "    Layer::A dimensions(" << A.getRows() << "," << A.getCols() << ")" << endl;
    return &A;
}

Matrix Layer::back_propagate(Matrix& A_prev) {
    // cout << "    Layer::Initialized back_propagate" << endl;
    // cout << "    Layer::Layer_size: " << layer_size << " Prev_size: " << prev_size << endl;
    if (type == "output") {
        // cout << "    Layer::Type Output" << endl;
        Matrix aux = (A*-1.0)+1.0;
        Matrix g_prime = A.mulElementWise(aux);
        dZ = dA.mulElementWise(g_prime);
    }
    else {
        // cout << "    Layer::Type Hidden" << endl;
        Matrix g_prime = A.relu_prime();
        // cout << "    Layer::g_prime dimensions(" << g_prime.getRows() << "," << g_prime.getCols() << ")" << endl;
        // cout << "    Layer::dA dimensions(" << dA.getRows() << "," << dA.getCols() << ")" << endl;
        dZ = dA.mulElementWise(g_prime);
    }
    // cout << "    Layer::Calculated dZ" << endl;
    double m = A_prev.getCols();
    Matrix A_prevT = A_prev.transpose();
    // cout << "    Layer::Transposed A_prev" << endl;
    dW = (dZ*A_prevT)/m;
    // cout << "    Layer::Calculated dW" << endl;
    db = (dZ.sum(1)/m).transpose();
    // cout << "    Layer::Calculated db" << endl;
    Matrix dA_prev = W.transpose()*dZ;
    // cout << "    Layer::dA_prev dimensions(" << dA_prev.getRows() << "," << dA_prev.getCols() << ")" << endl;
    return dA_prev;
}

void Layer::update_parameters(double learning_rate) {
    // cout << "    Layer::update_parameters" << endl;
    auto Waux = dW*learning_rate;
    auto baux = db*learning_rate;
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
