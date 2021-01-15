#include <iostream>
#include <vector>
#include <string>
// #include "activation_functions.hh"
#include "model.hh"
// #include "loss.hh"
#include "data_processing.hh"
#include "matrix.hh"

void test() {
    // Matrix m1(3,3,0.5);
    Matrix m2(2,2);
    Matrix m3(2,2,1.0);
    m2(0,1) = 1.0;
    m3(1,1) = 0.0;
    auto loss = (m2-m3).pow2Matrix();
    m2.printMatrix();
    m3.printMatrix();
    loss.printMatrix();
    Matrix aux = ((loss).sum(1))/(m2.getCols());
    aux.printMatrix();
    auto res = aux.sum()/aux.getRows();
    cout << res << endl;
    // m3(0,0) = -1;
    // m3(1,0) = 1;
    // m3(2,0) = 3;
    // m3(3) = 5;
    // Matrix m5(5,7, 0.5);
    // m4(1) = 1.5;
    // m4(2) = 20.5;
    // m4(3) = 40.5;
    // m4(4) = 50.5;
    // m2.printMatrix();
    // m3.printMatrix();
    // m6.printMatrix();
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
    vector<string> layers_type{ "relu", "relu", "softmax" };
    Model m(x_train, y_train, "cross_entropy", layers_dims, layers_type, 0.001, 20);
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
