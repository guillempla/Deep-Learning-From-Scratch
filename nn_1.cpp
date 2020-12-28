#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include "Neuron.hpp"

class NeuralNetwork
{
public:
    NeuralNetwork(const std::vector<unsigned> &, bool = false);
    void status();
    void setWeight(unsigned, unsigned, unsigned, double);
    void feedForward(const std::vector<double> &);
    void backPropagate(const std::vector<double> &);
    std::vector<double> getOutput();

private:
    std::vector<Layer> vecLayers;
    bool Bias;
};

#endif

