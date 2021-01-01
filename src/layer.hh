#ifndef LAYER_HH
#define LAYER_HH

#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.hh"

using namespace std;

class Layer {
    private:
        int type;       // 0: input layer; 1: hidden layer, 2: output layer
        unsigned layer_size;
        unsigned next_size;

        Matrix bias;    // matrix of (next_size,1)
        Matrix weights; // matrix of (next_size,actual_size)

        Matrix activation;
        Matrix activation_cache;

        Matrix weights_prime;   // matrix of weights derivatives

    public:
        //___________CONSTRUCTORS__________
        Layer(int type, unsigned layer_size, unsigned next_size);

        //___________SETTERS__________
        void set_type(int type);
        void set_layer_size(unsigned layer_size);
        void set_next_size(unsigned next_size);
        void set_bias(const Matrix& bias);
        void set_weights(const Matrix& weights);
        void set_weights_prime(const Matrix& weights_prime);

        void linear_activation_forward(const Matrix& );


        //___________GETTERS__________
        int get_type() const;
        Matrix get_bias() const;
        Matrix get_weights() const;
        Matrix get_weights_prime() const;
        Matrix get_activation() const;
        Matrix get_activation_cache() const;



};

#endif
