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
        int layer_size;
        int next_size;

        Matrix bias;    // matrix of (next_size,1)
        Matrix weights; // matrix of (next_size,actual_size)

        void initialize_bias();
        void initialize_weights();

    public:
        //___________CONSTRUCTORS__________
        Layer(int type, int layer_size, int next_size);

        //___________SETTERS__________
        void set_type(int type);
        void set_layer_size(int layer_size);
        void set_next_size(int next_size);
        void set_bias(const Matrix& bias);
        void set_weights(const Matrix& weights);


        //___________GETTERS__________
        int get_type() const;
        Matrix get_bias() const;
        Matrix get_weights() const;



};

#endif
