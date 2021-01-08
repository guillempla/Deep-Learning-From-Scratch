#ifndef MODEL_HH
#define MODEL_HH

#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.hh"
#include "layer.hh"

using namespace std;

class Model {
    private:
        Matrix X;
        Matrix Y;
        unsigned num_examples;
        vector<Layer> layers;
        float learning_rate;
        unsigned num_iter;

        void initialize_parameters(const vector<unsigned> layer_dims);

    public:
        //___________CONSTRUCTORS__________
        /*
        layer_dims: position i constains size of layer i
        */
        Model(const Matrix& X, const Matrix& Y, const vector<unsigned>& layers_dims, float learning_rate, unsigned num_iter);

        //___________SETTERS__________
        void feed_forward();
        void back_propagate();



};

#endif
