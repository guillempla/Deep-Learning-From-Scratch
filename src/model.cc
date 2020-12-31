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
void Model::feed_forward(const vector<double>& inputs) {

}

void Model::back_propagate(const vector<double>& predictions) {

}



//___________GETTERS__________

//___________PRIVATE__________
void Model::initialize_parameters(const vector<unsigned> layers_dims) {
    this->layers.reserve(layer_dims.size());
    for (int i = 0; i < layer_dims.size(); i++) {
        if (i == 0) {
            Layer l(type, layer_dims[i], layer_dims[i+1]);

        }
        else if (i == layer_dims.size()-1) {

        }
        else {

        }
    }
}
