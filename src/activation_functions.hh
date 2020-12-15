#ifndef ACTIVATION_FUNCTIONS_HH
#define ACTIVATION_FUNCTIONS_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Activation_functions {
    private:
        float steepness;

    public:
        //___________CONSTRUCTORS__________
        Activation_functions(const float steepness);

        //___________SETTERS__________
        void modify_steepness(const float steepness);


        //___________GETTERS__________

        // Return steepness private parameter
        float steepness() const;

        // Activation functions
        float sigmoid(const float z) const;
        float relu(const float z) const;

        // Dervatives of activation functions
        float sigmoid_prime(const float z) const;
        float relu_prime(const float z) const;

}

#endif
