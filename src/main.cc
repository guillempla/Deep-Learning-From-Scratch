#include <iostream>
#include <vector>
#include <string>
// #include "activation_functions.hh"
// #include "loss.hh"
#include "data_processing.hh"
#include "matrix.hh"

int main() {
    Matrix m1(3,3,0.5);
    Matrix m2(3,3,1.0);

    Matrix m3(1,3);
    m3(0) = 0;
    m3(1) = 1;
    m3(2) = 2;
    // cout << m3(3);

    auto m = m1*m2;

    m.printMatrix();

    m.sigmoid();
    m.printMatrix();

    m.relu();
    m.printMatrix();

    m.sigmoid_prime();
    m.printMatrix();

    m.relu_prime();
    m.printMatrix();
}
