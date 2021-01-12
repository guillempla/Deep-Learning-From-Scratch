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
    Matrix m2(5,7,false,1);
    Matrix m3(1,5, false, 1);
    m3.printMatrix();
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
    cout << "Started main" << endl;
    Data_processing d;
    Matrix y_test = d.read_test_labels();
    Matrix x_test = d.read_test_vectors();
    Matrix y_train = d.read_train_labels();
    Matrix x_train = d.read_train_vectors();

    cout << "Finished read data" << endl;

    vector<unsigned> layers_dims{ x_train.getRows(), 64, 10 };

    Model m(x_train, y_train, layers_dims, 0.0075, 10);

    Matrix costs = m.train();

    Matrix predictions = m.predict(x_test);

    d.write_predictions(predictions);
}

int main() {
    // test();
    real_main();
}
