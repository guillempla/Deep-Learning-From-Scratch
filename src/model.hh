#ifndef MODEL_HH
#define MODEL_HH

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include "matrix.hh"
#include "layer.hh"
#include "loss.hh"
#include "data_processing.hh"

using namespace std;

class Model {
    private:
        Matrix X;
        Matrix Y;
        string loss;
        vector<Layer> layers;
        double learning_rate;
        int epochs;
        int batch_size;
        int num_batches;
        double lambd;

        void initialize_layers(const vector<unsigned>& layer_dims, const vector<string>& layers_type, unsigned num_examples);

        Matrix get_batch_x(int i, int size);
        Matrix get_batch_y(int i, int size);

        Matrix* get_previous_activation(Matrix& input, unsigned i);

        double compute_cost(Matrix& output);
        double compute_accuracy(Matrix& input, Matrix& output);
        Matrix derivate_cost(Matrix& output);        // Calculate dAL

        double l2_regularization() const;
        static double frobenius_norm(Matrix& W);

    public:
        //___________CONSTRUCTORS__________
        /*
        layer_dims: position i contains size of layer i
        */
        Model(const Matrix& X, const Matrix& Y, const string& loss, const vector<unsigned>& layers_dims, const vector<string>& layers_type, double learning_rate, int epochs, int batch_size, double lambd);

        //___________SETTERS__________
        void train();
        Matrix predict(Matrix& input);

        void feed_forward(Matrix& input);
        void back_propagate(Matrix& input, Matrix& output);
        void update_parameters();
};

#endif
