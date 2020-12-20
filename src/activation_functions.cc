#include "activation_functions.hh"

//___________CONSTRUCTORS__________
Activation_functions::Activation_functions() {
    this->steepness = 1;
}

Activation_functions::Activation_functions(double steepness) {
    this->steepness = steepness;
}

//___________SETTERS__________
void Activation_functions::modify_steepness(double steepness) {
    this->steepness = steepness;
}


//___________GETTERS__________
double Activation_functions::get_steepness() const {
    return this->steepness;
}


// Activation functions
double Activation_functions::sigmoid(const double z) const {
    return 1.0 / (1 + exp(-z*this->steepness));
}

double Activation_functions::relu(const double z) const {
    return max(z, 0.0);
}


// Dervatives of activation functions
double Activation_functions::sigmoid_prime(const double z) const {
    return this->sigmoid(z) / (1 - this->sigmoid(z*this->steepness));
}

double Activation_functions::relu_prime(const double z) const {
    if (z > 0)
        return 1;
    else if (z < 0)
        return 0;
}
