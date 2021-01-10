#ifndef LAYER_HH
#define LAYER_HH

#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.hh"

using namespace std;

class Layer {
    private:
        bool type;                  // false: hidden layer; true: output layer
        unsigned layer_size;
        unsigned prev_size;
        unsigned num_examples;

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
        Layer(bool type, unsigned num_examples, unsigned layer_size, unsigned prev_size);

        //___________SETTERS__________
        void set_type(bool type);
        void set_layer_size(unsigned layer_size);
        void set_prev_size(unsigned prev_size);
        void set_bias(const Matrix& b);
        void set_weights(const Matrix& W);
        void set_weights_prime(const Matrix& dW);

        /*
        A_prev: activations of previous layer
        */
        Matrix* feed_forward(Matrix& A_prev);
        Matrix* back_propagate(Matrix& A_prev, Matrix& dA_prev);

        //___________GETTERS__________
        bool get_type() const;
        Matrix get_bias() const;
        Matrix get_weights() const;
        Matrix get_weights_prime() const;
        Matrix* get_activation();



};

#endif
