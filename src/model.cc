#include "model.hh"

//___________CONSTRUCTORS__________
Model::Model(const Matrix& X, const Matrix& Y, const vector<unsigned>& layers_dims, double learning_rate, unsigned num_iter) {
    // cout << "Model::Initializing model" << endl;
    this->X = X;
    this->Y = Y;
    this->learning_rate = learning_rate;
    this->num_iter = num_iter;
    this->initialize_layers(layers_dims, X.getCols());
}


//___________SETTERS__________
void Model::train() {
    for (unsigned i = 0; i < num_iter; i++) {
        cout << "Iteration: " << i << endl;
        feed_forward();
        Matrix* AL = layers[layers.size()-1].get_activation();
        cout << "Cost: " << Loss::mean_square(Y, *AL) << endl;
        back_propagate();
        update_parameters();
    }
}

void Model::feed_forward() {
    // cout << "Model::feed_forward" << endl;
    Matrix *A_prev = &X;
    for (auto& layer: this->layers)
        A_prev = layer.feed_forward(*A_prev);
}

void Model::back_propagate() {
    // cout << "Model::back_propagate" << endl;
    Matrix* A = layers[layers.size()-1].get_activation();
    // cout << "Model::get_activation" << endl;
    // cout << "Model::Y dimensions(" << Y.getRows() << "," << Y.getCols() << ")" << endl;
    // cout << "Model::A dimensions(" << A->getRows() << "," << A->getCols() << ")" << endl;
    Matrix dA = Loss::mean_square_prime(Y, *A);
    // cout << "Model::dA calculated" << endl;
    for (int i = layers.size()-1; i >= 0; i--) {
        auto& layer = this->layers[i];
        layer.set_activation_gradient(dA);
        // cout << "Model::set_activation_gradient" << endl;
        Matrix* A_prev = get_previous_activation(i);
        // cout << "Model::get_previous_activation" << endl;
        dA = layer.back_propagate(*A_prev);
    }
}

void Model::update_parameters() {
    // cout << "Model::update_parameters" << endl;
    for (auto& layer: layers)
        layer.update_parameters(learning_rate);
    // cout << "Model::finished updating parameters" << endl;
}



//___________GETTERS__________

//___________PRIVATE__________
void Model::initialize_layers(const vector<unsigned> layers_dims, unsigned num_examples) {
    // cout << "Model::Initializing layers" << endl;
    this->layers.reserve(layers_dims.size()-1);
    string type = "hidden";
    for (int i = 1; i < layers_dims.size(); i++) {
        // cout << "    Model::Iteration: " << i << endl;
        if (i == layers_dims.size()-1)
            type = "output";
        // cout << "    Model::Type: " << type << endl;
        Layer l = Layer(type, num_examples, layers_dims[i], layers_dims[i-1]);
        // cout << "    Model::Finished init Layer" << endl;
        this->layers.push_back(l);
        // cout << "    Model::Finished init Layer(" << i-1 << ")" << endl;
    }
}

Matrix* Model::get_previous_activation(int i) {
    if (i == 0)
        return &X;
    return layers[i-1].get_activation();
}
