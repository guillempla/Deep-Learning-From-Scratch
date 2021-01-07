#ifndef ACTIVATION_HH
#define ACTIVATION_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Activation {
    private:
        double steepness;

    public:
        //___________CONSTRUCTORS__________
        Activation();
        Activation(double steepness);

        //___________SETTERS__________
        void modify_steepness(double steepness);


        //___________GETTERS__________

        // Return steepness private parameter
        double get_steepness() const;

        // Activation functions
        static double sigmoid(const double z);
        static double relu(const double z);

        // Derivatives of activation functions
        static double sigmoid_prime(const double z);
        static double relu_prime(const double z);

};

#endif
