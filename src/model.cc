#include "model.hh"

//___________CONSTRUCTORS__________
Model::Model(const Matrix& X, const Matrix& Y, const string& loss, const vector<unsigned>& layers_dims, const vector<string>& layers_type, double learning_rate, int epochs, int batch_size, double lambd) {
    this->X = X;
    this->Y = Y;

    random_device rd;
    int seed = rd();
    this->X.shuffleMatrix(seed);
    this->Y.shuffleMatrix(seed);

    this->learning_rate = learning_rate;
    this->epochs = epochs;
    this->batch_size = batch_size;
    this->loss = loss;
    this->lambd = lambd;

    this->num_batches = ceil(X.getCols()/batch_size);

    this->initialize_layers(layers_dims, layers_type, num_batches);
}


//___________SETTERS__________
void Model::train() {
    // first three batches used for testing accuracy
    Matrix x_dev = get_batch_x(0, 3*batch_size);
    Matrix y_dev = get_batch_y(0, 3*batch_size);

    for (int i = 0; i < epochs; i++) {
        for (int j = 3; j < num_batches; j++) {
            Matrix batch_x = get_batch_x(j, batch_size);
            Matrix batch_y = get_batch_y(j, batch_size);

            feed_forward(batch_x);
            //if (i % 1 == 0 && j % 200 == 0) {
            //    cout << "Epoch: " << i << " Batch: " << j << endl;
            //    cout << "Cost: " << compute_cost(batch_y) << endl;
            //}
            back_propagate(batch_x, batch_y);
            update_parameters();
        }
        cout << "Epoch: " << i << " ";
        cout << "Accuracy: " << compute_accuracy(x_dev, y_dev) << endl;
    }
}

Matrix Model::predict(Matrix& input) {
    Matrix A_prev = input;
    for (auto& layer: this->layers)
        A_prev = layer.predict(A_prev);
    return A_prev;
}

void Model::feed_forward(Matrix& input) {
    Matrix* A_prev = &input;
    for (auto& layer: this->layers)
        A_prev = layer.feed_forward(*A_prev);
}

void Model::back_propagate(Matrix& input, Matrix& output) {
    Matrix dA = derivate_cost(output);
    for (int i = (int)layers.size()-1; i >= 0; i--) {
        auto& layer = this->layers[i];
        layer.set_activation_gradient(dA);
        Matrix* A_prev = get_previous_activation(input, i);
        dA = layer.back_propagate(*A_prev, lambd);
    }
}

void Model::update_parameters() {
    for (auto& layer: layers)
        layer.update_parameters(learning_rate);
}

double Model::compute_cost(Matrix& output) {
    Matrix* AL = layers[layers.size()-1].get_activation();
    if (loss == "mean_square")
        return Loss::mean_square(output, *AL);
    else if (loss == "cross_entropy")
        return Loss::cross_entropy(output, *AL);
    else if (loss == "binary_cross_entropy")
        return Loss::binary_cross_entropy(output, *AL) + (lambd/(2*output.getCols()))*l2_regularization();
    else
        throw invalid_argument("ERROR compute_cost: Wrong error function!");
}

double Model::compute_accuracy(Matrix& input, Matrix& output) {
    Matrix predictions = predict(input);

    Matrix y_hat = Data_processing::convert_binary_matrix(predictions);
    Matrix y = Data_processing::convert_binary_matrix(output);

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

Matrix Model::get_batch_x(int i, int size) {
    int start = i*size;
    int finish = min(start+size, (int)X.getCols());
    Matrix batch_x(X.getRows(), finish-start);
    batch_x.copyFragment(X, 1, start);
    return batch_x;
}
Matrix Model::get_batch_y(int i, int size) {
    int start = i*size;
    int finish = min(start+size, (int)Y.getCols());
    Matrix batch_y(Y.getRows(), finish-start);
    batch_y.copyFragment(Y, 1, start);
    return batch_y;
}

Matrix* Model::get_previous_activation(Matrix& input, unsigned i) {
    return i == 0 ? &input : layers[i - 1].get_activation();
}

Matrix Model::derivate_cost(Matrix& output) {
    Matrix* AL = layers[layers.size() - 1].get_activation();
    Matrix dA(AL->getRows(), AL->getCols());
    if (loss == "mean_square")
        dA = Loss::mean_square_prime(output, *AL);
    else if (loss == "cross_entropy")
        dA = Loss::cross_entropy_prime(output, *AL);
    else if (loss == "binary_cross_entropy")
        dA = Loss::binary_cross_entropy_prime(output, *AL);
    else
        throw invalid_argument("ERROR derivate_cost: Wrong error function!");
    return dA;
}

double Model::l2_regularization() const {
    double sum = 0.0;
    for (auto layer: layers) {
        Matrix* W = layer.get_weights();
        sum += frobenius_norm(*W);
    }
    return sum;
}

double Model::frobenius_norm(Matrix& W) {
    return W.square().sum(1).sum();
}

