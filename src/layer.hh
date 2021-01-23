#ifndef LAYER_HH
#define LAYER_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include "matrix.hh"

using namespace std;

class Layer {
    private:
        string type;                // activation function type [relu, sigmoid, softmax]

        Matrix b;                   // matrix of bias (1,layer_size)
        Matrix W;                   // matrix of weights (layer_size,prev_size)

        Matrix Z;                   // matrix of inner potential (layer_size,num_examples)
        Matrix A;                   // matrix of activation (layer_size,num_examples)

        Matrix db;                  // matrix of bias derivatives (1,layer_size)
        Matrix dW;                  // matrix of weights derivatives (layer_size,prev_size)

        Matrix dZ;                  // matrix of potential derivatives (layer_size,num_examples)
        Matrix dA;                  // matrix of activation derivatives (layer_size,num_examples)

        void activation_backward(); // Calculate dZ

    public:
        //___________CONSTRUCTORS__________
        Layer(string type, unsigned num_examples, unsigned layer_size, unsigned prev_size);

        //___________SETTERS__________
        void set_activation_gradient(Matrix& dA);

        // A_prev: activations of previous layer
        Matrix predict(Matrix& A_prev);
        Matrix* feed_forward(Matrix& A_prev);
        Matrix back_propagate(Matrix& A_prev, double lambd);
        void update_parameters(double learning_rate);

        Matrix *get_weights();
        Matrix* get_activation();
};

#endif
