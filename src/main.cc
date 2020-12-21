#include <iostream>
#include <vector>
#include <string>
#include "activation_functions.hh"
#include "loss_functions.hh"
#include "data_processing.hh"
#include "matrix.hh"

int main() {
    Data_processing data;

    auto vectors = data.read_train_vectors();
    cout << "finished reading";

    cout << vectors.getRows() << " " << vectors.getCols();
}
