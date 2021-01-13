#ifndef LAYER_HH
#define LAYER_HH

#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <stdexcept>
#include "matrix.hh"

using namespace std;

class Layer {
    private:
        string type;                // activation function type [relu, sigmoid, softmax]
        unsigned layer_size;        // number of neurons of the layer
        unsigned prev_size;         // number of neurons of the previous layer
        unsigned num_examples;      // number of examples of the input set

        Matrix b;                   // matrix of bias (1,layer_size)
        Matrix W;                   // matrix of weights (layer_size,prev_size)

        Matrix Z;                   // matrix of inner potential (layer_size,num_examples)
        Matrix A;                   // matrix of activation (layer_size,num_examples)

        Matrix db;                  // matrix of bias derivatives (1,layer_size)
        Matrix dW;                  // matrix of weights derivatives (layer_size,prev_size)

        Matrix dZ;                  // matrix of potential derivatives (layer_size,num_examples)
        Matrix dA;                  // matrix of activation derivatives (layer_size,num_examples)

    public:
        //___________CONSTRUCTORS__________
        Layer(string type, unsigned num_examples, unsigned layer_size, unsigned prev_size);

        //___________SETTERS__________
        void set_type(string type);
        void set_layer_size(unsigned layer_size);
        void set_prev_size(unsigned prev_size);
        void set_bias(Matrix& b);
        void set_weights(Matrix& W);
        void set_weights_gradient(Matrix& dW);
        void set_activation_gradient(Matrix& dA);

        // A_prev: activations of previous layer
        Matrix predict(Matrix& A_prev);
        Matrix* feed_forward(Matrix& A_prev);
        Matrix back_propagate(Matrix& A_prev);
        void update_parameters(double learning_rate);

        //___________GETTERS__________
        string get_type() const;
        Matrix get_bias() const;
        Matrix get_weights() const;
        Matrix get_weights_gradient() const;
        Matrix* get_activation();
        Matrix* get_activation_gradient();



};

#endif
