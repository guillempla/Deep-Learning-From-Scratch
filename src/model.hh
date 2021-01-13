#ifndef MODEL_HH
#define MODEL_HH

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include "matrix.hh"
#include "layer.hh"
#include "loss.hh"

using namespace std;

class Model {
    private:
        Matrix X;
        Matrix Y;
        vector<Layer> layers;
        double learning_rate;
        unsigned num_iter;

        void initialize_layers(const vector<unsigned> layer_dims, const vector<string>& layers_type, unsigned num_examples);
        Matrix* get_previous_activation(int i);

    public:
        //___________CONSTRUCTORS__________
        /*
        layer_dims: position i constains size of layer i
        */
        Model(const Matrix& X, const Matrix& Y, const vector<unsigned>& layers_dims, const vector<string>& layers_type, double learning_rate, unsigned num_iter);

        //___________SETTERS__________
        Matrix train();
        Matrix predict(Matrix& input);

        void feed_forward();
        void back_propagate();
        void update_parameters();
        double compute_cost();



};

#endif
