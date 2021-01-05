#include "model.hh"

//___________CONSTRUCTORS__________
Model::Model(const Matrix& X, const Matrix& Y, const vector<unsigned>& layers_dims, float learning_rate, unsigned num_iter) {
    this->X = X;
    this->Y = Y.transpose();
    this->learning_rate;
    this->num_iter;
    this->initialize_parameters(layers_dims);
}


//___________SETTERS__________
void Model::feed_forward() {
    Matrix *A_prev = &X;
    for (auto& layer: this->layers) {
        layer.feed_forward(*A_prev);
        A_prev = layer.get_activation();
    }
}

void Model::back_propagate() {
    Matrix *A_prev = this->layers[layers.size()-1].get_activation();
    Matrix dA_prev = Y - *A_prev;
    for (int i = layers.size()-1; i >= 0; i--) {
        auto& layer = this->layers[i];
        dA_prev = layer.back_propagate(dA_prev);
    }
}



//___________GETTERS__________

//___________PRIVATE__________
void Model::initialize_parameters(const vector<unsigned> layers_dims) {
    this->layers.reserve(layers_dims.size()-1);
    int type = 0;
    for (int i = 1; i < layers_dims.size(); i++) {
        if (i == layers_dims.size()-1)
            type = 2;
        this->layers[i-1] = Layer(type, layers_dims[i], layers_dims[i-1]);
        if (i == 1)
            type = 1;
    }
}
