#include <iostream>
#include <vector>
#include <string>
// #include "activation_functions.hh"
// #include "loss.hh"
#include "data_processing.hh"
#include "matrix.hh"

void test() {
    // Matrix m1(3,3,0.5);
    // Matrix m2(3,3,1.0);
    //
    // Matrix m3(1,3);
    // m3(0) = 0;
    // m3(1) = 1;
    // m3(2) = 2;
    // // cout << m3(3);
    //
    // auto m = m1*m2;
    //
    // m.printMatrix();
    //
    // m.sigmoid();
    // m.printMatrix();
    //
    // m.relu();
    // m.printMatrix();
    //
    // m.sigmoid_prime();
    // m.printMatrix();
    //
    // m.relu_prime();
    // m.printMatrix();

    Matrix m5(5,7, 0.5);

    Matrix m4(1, 5, 0.5);
    m4(1) = 1.5;
    m4(2) = 20.5;
    m4(3) = 40.5;
    m4(4) = 50.5;
    // m4.printMatrix();
    // m5.printMatrix();
    Matrix m6(5,7);
    m6 = m5+m4;
    m6.printMatrix();
}

void real_main() {

}

int main() {
    test();
    // real_main();
}
