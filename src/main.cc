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
    // m3(0,0) = -1;
    // m3(1,0) = 1;
    // m3(2,0) = 3;
    // m3(3) = 5;
    // Matrix m5(5,7, 0.5);
    // Matrix m4(1, 5, 0.5);
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
    auto test_labels = d.read_test_labels();
    auto test_vectors = d.read_test_vectors();
    auto Y = d.read_train_labels();
    auto X = d.read_train_vectors();

    cout << "Finished read data" << endl;

    vector<unsigned> layers_dims{ X.getRows(), 2, 7, 2 };

    Model m(X, Y, layers_dims, 30, 1000);

    m.feed_forward();

    // d.write_predictions(predictions);
}

int main() {
    // test();
    real_main();
}
