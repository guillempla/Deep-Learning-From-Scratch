#include "activation.hh"

//___________CONSTRUCTORS__________
Activation::Activation() {
    this->steepness = 1;
}

Activation::Activation(double steepness) {
    this->steepness = steepness;
}

//___________SETTERS__________
void Activation::modify_steepness(double steepness) {
    this->steepness = steepness;
}


//___________GETTERS__________
double Activation::get_steepness() const {
    return this->steepness;
}


// Activation functions
double Activation::sigmoid(const double z) {
    return 1.0 / (1 + exp(-z));
}

double Activation::relu(const double z) {
    return max(z, 0.0);
}

// TODO: Check argument, should I pass Z or activation(Z)?
// Derivatives of activation functions
double Activation::sigmoid_prime(const double z) {
    return Activation::sigmoid(z) / (1 - Activation::sigmoid(z));
}

double Activation::relu_prime(const double z) {
    if (z > 0)
        return 1;
    return 0;
}
