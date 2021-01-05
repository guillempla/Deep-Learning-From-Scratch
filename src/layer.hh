#ifndef LAYER_HH
#define LAYER_HH

#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.hh"
#include "activation_functions.hh"

using namespace std;

class Layer {
    private:
        int type;                   // 0: input layer; 1: hidden layer, 2: output layer
        unsigned layer_size;
        unsigned prev_size;

        Matrix bias;                // matrix of (layer_size,1)
        Matrix weights;             // matrix of (layer_size,prev_size)

        Matrix potential;           // matrix of inner potential
        Matrix activation;          // matrix of activation

        Matrix weights_prime;       // matrix of weights derivatives
        Matrix bias_prime;          // matrix of bias derivatives

        Matrix potential_prime;     // matrix of potential derivatives
        Matrix activation_prime;    // matrix of activation derivatives

        void forward(Matrix& A_prev);

    public:
        //___________CONSTRUCTORS__________
        Layer(int type, unsigned layer_size, unsigned prev_size);

        //___________SETTERS__________
        void set_type(int type);
        void set_layer_size(unsigned layer_size);
        void set_prev_size(unsigned prev_size);
        void set_bias(const Matrix& bias);
        void set_weights(const Matrix& weights);
        void set_weights_prime(const Matrix& weights_prime);

        /*
        A_prev: activations of previous layer
        */
        void feed_forward(Matrix& A_prev);
        Matrix back_propagate(Matrix& dA_prev);

        //___________GETTERS__________
        int get_type() const;
        Matrix get_bias() const;
        Matrix get_weights() const;
        Matrix get_weights_prime() const;
        Matrix* get_activation();



};

#endif
