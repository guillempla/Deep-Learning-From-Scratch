#include "activation_functions.hh"

//___________CONSTRUCTORS__________
Activation_functions::Activation_functions() {
    this->steepness = 1;
}

Activation_functions::Activation_functions(const float steepness) {
    this->steepness = steepness;
}

//___________SETTERS__________
void Activation_functions::modify_steepness(const float steepness) {
    this->steepness = steepness;
}


//___________GETTERS__________
float Activation_functions::get_steepness() const {
    return this->steepness;
}


// Activation functions
float Activation_functions::sigmoid(const float z) const {
    return 1.0 / (1 + exp(-z*this->steepness));
}

float Activation_functions::relu(const float z) const {
    return max(z, 0);
}


// Dervatives of activation functions
float Activation_functions::sigmoid_prime(const float z) const {
    return this->sigmoid(z) / (1 - this->sigmoid(z*this->steepness));
}

float Activation_functions::relu_prime(const float z) const {
    if (z > 0)
        return 1;
    else if (z < 0)
        return 0;
}
