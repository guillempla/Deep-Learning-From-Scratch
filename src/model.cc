#include "model.hh"

//___________CONSTRUCTORS__________
Model::Model(const Matrix& X, const Matrix& Y, const string& loss, const vector<unsigned>& layers_dims, const vector<string>& layers_type, double learning_rate, unsigned num_iter) {
    // cout << "Model::Initializing model" << endl;
    this->X = X;
    this->Y = Y;
    this->learning_rate = learning_rate;
    this->num_iter = num_iter;
    this->loss = loss;
    this->initialize_layers(layers_dims, layers_type, X.getCols());
}


//___________SETTERS__________
Matrix Model::train() {
    Matrix costs(1, num_iter);
    for (unsigned i = 0; i < num_iter; i++) {
        feed_forward();
        costs(i) = compute_cost();
        if (i % 10 == 0) {
            cout << "Iteration: " << i << endl;
            cout << "Cost: " << costs(i) << endl;
        }
        back_propagate();
        update_parameters();
    }
    return costs;
}

Matrix Model::predict(Matrix& input) {
    // // cout << "Model::predict" << endl;
    Matrix A_prev = input;
    for (auto& layer: this->layers)
        A_prev = layer.predict(A_prev);
    return A_prev;
}

void Model::feed_forward() {
    // cout << "Model::feed_forward" << endl;
    Matrix* A_prev = &X;
    for (auto& layer: this->layers)
        A_prev = layer.feed_forward(*A_prev);
}

void Model::back_propagate() {
    // cout << "Model::back_propagate" << endl;
    Matrix* A = layers[layers.size()-1].get_activation();
    Matrix dA(A->getRows(), A->getCols());
    if (loss == "mean_square")
        dA = Loss::mean_square_prime(Y, *A);
    else if (loss == "cross_entropy")
        dA = Loss::cross_entropy_prime(Y, *A);
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

double Model::compute_cost() {
    Matrix* AL = layers[layers.size()-1].get_activation();
    if (loss == "mean_square")
        return Loss::mean_square(Y, *AL);
    else if (loss == "cross_entropy")
        return Loss::cross_entropy(Y, *AL);
    else
        return -1.0;
}



//___________GETTERS__________

//___________PRIVATE__________
void Model::initialize_layers(const vector<unsigned> layers_dims, const vector<string>& layers_type, unsigned num_examples) {
    // cout << "Model::Initializing layers" << endl;
    this->layers.reserve(layers_dims.size()-1);
    for (int i = 1; i < layers_dims.size(); i++) {
        // cout << "    Model::Iteration: " << i << endl;
        Layer l = Layer(layers_type[i-1], num_examples, layers_dims[i], layers_dims[i-1]);
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
