#include <iostream>
#include <vector>
#include <string>
#include "model.hh"
#include "data_processing.hh"
#include "matrix.hh"

void test() {
    Matrix m1(256,736);
    for (unsigned i = 0; i < m1.getRows(); i++) {
        for (unsigned j = 0; j < m1.getCols(); j++) {
            m1(i,j) = i+j;
        }
    }
    cout << "m1 "; m1.printMatrix();
    Matrix m2(736,60000,0.5);
    cout << "m2 "; m2.printMatrix();
    auto m3 = m1*m2;
    cout << "m3 "; m3.printMatrix();
}

void real_main() {
    cout << "Started ./network" << endl;

    Data_processing d;
    Matrix y_test = d.read_test_labels();
    Matrix x_test = d.read_test_vectors();
    Matrix y_train = d.read_train_labels();
    Matrix x_train = d.read_train_vectors();
    cout << "Finished reading data" << endl;

    vector<unsigned> layers_dims{ x_train.getRows(), 256, 64, 10 };
    vector<string> layers_type{ "relu", "relu", "sigmoid" };
    Model m(x_train, y_train, "binary_cross_entropy", layers_dims, layers_type, 0.001, 20, 0.1);
    cout << "Finished initalizing" << endl;

    Matrix costs = m.train();
    cout << "Finished training" << endl;

    Matrix predictions = m.predict(x_test);
    cout << "Finished predicting" << endl;

    d.write_predictions(predictions);
    cout << "Finished writing predictions" << endl;
}

int main() {
    // test();
    real_main();
}
