#include "model.hh"

//___________CONSTRUCTORS__________
Model::Model(const Matrix& X, const Matrix& Y, const vector<unsigned>& layers_dims, float learning_rate, unsigned num_iter) {
    cout << "Model::Initializing model" << endl;
    this->X = X;
    this->Y = Y;
    this->learning_rate;
    this->num_iter;
    this->initialize_layers(layers_dims, X.getCols());
}


//___________SETTERS__________
void Model::feed_forward() {
    cout << "Model::feed_forward" << endl;
    Matrix *A_prev = &X;
    for (auto& layer: this->layers)
        A_prev = layer.feed_forward(*A_prev);
}

void Model::back_propagate() {
    Matrix* A = layers[layers.size()-1].get_activation();
    Matrix dA = Loss::mean_square_prime(Y, *A);
    for (int i = layers.size()-1; i >= 0; i--) {
        auto& layer = this->layers[i];
        layer.set_activation_gradient(dA);
        Matrix* A_prev = get_previous_activation(i);
        Matrix dA = layer.back_propagate(*A_prev);
    }
}



//___________GETTERS__________

//___________PRIVATE__________
void Model::initialize_layers(const vector<unsigned> layers_dims, unsigned num_examples) {
    cout << "Model::Initializing layers" << endl;
    this->layers.reserve(layers_dims.size()-1);
    string type = "hidden";
    for (int i = 1; i < layers_dims.size(); i++) {
        cout << "    Model::Iteration: " << i << endl;
        if (i == layers_dims.size()-1)
            type = "output";
        cout << "    Model::Type: " << type << endl;
        Layer l = Layer(type, num_examples, layers_dims[i], layers_dims[i-1]);
        cout << "    Model::Finished init Layer" << endl;
        this->layers.push_back(l);
        cout << "    Model::Finished init Layer(" << i-1 << ")" << endl;
    }
}

Matrix* Model::get_previous_activation(int i) {
    if (i == 0)
        return &X;
    return layers[i-1].get_activation();
}
