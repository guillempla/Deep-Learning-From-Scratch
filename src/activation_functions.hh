#ifndef ACTIVATION_FUNCTIONS_HH
#define ACTIVATION_FUNCTIONS_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Activation_functions {
    private:
        double steepness;

    public:
        //___________CONSTRUCTORS__________
        Activation_functions();
        Activation_functions(double steepness);

        //___________SETTERS__________
        void modify_steepness(double steepness);


        //___________GETTERS__________

        // Return steepness private parameter
        double get_steepness() const;

        // Activation functions
        double sigmoid(const double z) const;
        double relu(const double z) const;

        // Dervatives of activation functions
        double sigmoid_prime(const double z) const;
        double relu_prime(const double z) const;

};

#endif
