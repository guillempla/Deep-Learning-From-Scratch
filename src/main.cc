#include <iostream>
#include <vector>
#include <string>
#include "model.hh"
#include "data_processing.hh"
#include "matrix.hh"

int main() {
    cout << "Started ./network" << endl;

    Data_processing d;
    Matrix y_test = d.read_test_labels();
    Matrix x_test = d.read_test_vectors();
    Matrix y_train = d.read_train_labels();
    Matrix x_train = d.read_train_vectors();
    cout << "Finished reading data" << endl;

    vector<unsigned> layers_dims{ x_train.getRows(), 256, 64, 10 };
    vector<string> layers_type{ "relu", "relu", "sigmoid" };
    Model m(x_train, y_train, "binary_cross_entropy", layers_dims, layers_type, 0.001, 35, 32, 0.7);
    cout << "Finished initalizing" << endl;

    m.train();
    cout << "Finished training" << endl;

    Matrix predictions = m.predict(x_test);
    cout << "Finished predicting" << endl;

    d.write_predictions(predictions);
    cout << "Finished writing predictions" << endl;
}
