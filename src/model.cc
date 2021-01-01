#include "model.hh"

//___________CONSTRUCTORS__________
Model::Model(const Matrix& X, const Matrix& Y, const vector<unsigned>& layers_dims, float learning_rate, unsigned num_iter) {
    this->X = X;
    this->Y = Y;
    this->learning_rate;
    this->num_iter;
    this->initialize_parameters(layers_dims);
}


//___________SETTERS__________
void Model::feed_forward() {
    // Matrix A_prev = &X;
    // for (auto& layer: this->layers) {
    //     layer.linear_activation_forward();
    // }
}

void Model::back_propagate() {

}



//___________GETTERS__________

//___________PRIVATE__________
void Model::initialize_parameters(const vector<unsigned> layers_dims) {
    this->layers.reserve(layers_dims.size()-1);
    int type = 0;
    for (int i = 0; i < this->layers.size(); i++) {
        if (i == this->layers.size()-1)
            type = 2;
        this->layers[i] = Layer(type, layers_dims[i], layers_dims[i+1]);
        if (i == 0)
            type = 1;
    }
}
