#include <iostream>
#include <vector>
#include <string>
#include "activation_functions.hh"
#include "loss_functions.hh"
#include "data_processing.hh"
#include "matrix.hh"

int main() {
    Matrix m1(3000,3000,4.5);
    Matrix m2(3000,3000,9.0);

    auto m = m1*m2;
    // m.printMatrix();
}
