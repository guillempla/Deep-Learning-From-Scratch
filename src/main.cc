#include <iostream>
#include <vector>
#include <string>
#include "activation_functions.hh"
#include "loss_functions.hh"
#include "data_processing.hh"
#include "matrix.hh"

int main() {
    Data_processing data;

    auto labels = data.read_test_labels();

    for (auto label: labels)
        cout << label << "\n";
}
