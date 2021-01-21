#include "model.hh"

//___________CONSTRUCTORS__________
Model::Model(const Matrix& X, const Matrix& Y, const string& loss, const vector<unsigned>& layers_dims, const vector<string>& layers_type, double learning_rate, unsigned epochs, unsigned C) {
    // cout << "Model::Initializing model" << endl;
    this->X = X;
    this->Y = Y;
    this->learning_rate = learning_rate;
    this->epochs = epochs;
    this->loss = loss;
    this->C = C;
    this->initialize_layers(layers_dims, layers_type, X.getCols());
}


//___________SETTERS__________
Matrix Model::train() {
    Matrix costs(1, epochs);
    for (unsigned i = 0; i < epochs; i++) {
        feed_forward();
        costs(i) = compute_cost();
        if (i % 1 == 0) {
            cout << "Iteration: " << i << endl;
            cout << "Cost: " << costs(i) << endl;
            cout << "Accuracy: " << compute_accuracy() << endl;
        }
        back_propagate();
        update_parameters();
    }
    return costs;
}

Matrix Model::predict(Matrix& input) {
    // cout << "Model::predict" << endl;
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
    Matrix dA = derivate_cost();
    for (int i = (int)layers.size()-1; i >= 0; i--) {
        auto& layer = this->layers[i];
        layer.set_activation_gradient(dA);
        Matrix* A_prev = get_previous_activation(i);
        dA = layer.back_propagate(*A_prev);
    }
}

void Model::update_parameters() {
    // cout << "Model::update_parameters" << endl;
    for (auto& layer: layers)
        layer.update_parameters(learning_rate);
}

double Model::compute_cost() {
    Matrix* AL = layers[layers.size()-1].get_activation();
    if (loss == "mean_square")
        return Loss::mean_square(Y, *AL);
    else if (loss == "cross_entropy")
        return Loss::cross_entropy(Y, *AL);
    else if (loss == "binary_cross_entropy")
        return Loss::binary_cross_entropy(Y, *AL);
    else
        throw invalid_argument("ERROR compute_cost: Wrong error function!");
}

double Model::compute_accuracy() {
    Matrix* AL = layers[layers.size()-1].get_activation();
    Matrix y_hat = Data_processing::convert_binary_matrix(*AL);
    Matrix y = Data_processing::convert_binary_matrix(Y);

    double hits = 0.0;
    #pragma omp parallel for reduction (+:hits) num_threads(16)
    for (unsigned i = 0; i < y.getCols(); i++) {
        if (y_hat(i) == y(i))
            hits++;
    }
    return hits/((double)y.getCols());
}



//___________GETTERS__________

//___________PRIVATE__________
void Model::initialize_layers(const vector<unsigned>& layers_dims, const vector<string>& layers_type, unsigned num_examples) {
    this->layers.reserve(layers_dims.size()-1);
    for (int i = 1; i < layers_dims.size(); i++) {
        Layer l = Layer(layers_type[i-1], num_examples, layers_dims[i], layers_dims[i-1]);
        this->layers.push_back(l);
    }
}

Matrix* Model::get_previous_activation(unsigned i) {
    return i == 0 ? &X : layers[i - 1].get_activation();
}

Matrix Model::derivate_cost() {
    Matrix* AL = layers[layers.size() - 1].get_activation();
    Matrix dA(AL->getRows(), AL->getCols());
    if (loss == "mean_square")
        dA = Loss::mean_square_prime(Y, *AL);
    else if (loss == "cross_entropy")
        dA = Loss::cross_entropy_prime(Y, *AL);
    else if (loss == "binary_cross_entropy")
        dA = Loss::binary_cross_entropy_prime(Y, *AL);
    else
        throw invalid_argument("ERROR derivate_cost: Wrong error function!");
    return dA;
}
