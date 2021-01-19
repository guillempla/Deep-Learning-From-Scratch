#ifndef ACTIVATION_HH
#define ACTIVATION_HH

#include <iostream>
#include <vector>
#include <cmath>

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
        static double sigmoid(double z);
        static double relu(double z);

        // Derivatives of activation functions
        static double sigmoid_prime(double z);
        static double relu_prime(double z);

};

#endif
