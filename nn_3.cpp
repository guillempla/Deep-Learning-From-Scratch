#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <random>

class Neuron;
typedef std::vector<Neuron> Layer;

struct Axon
{
    double weight;
    double deltaWeight; 
    Axon(double weight = 0): weight(weight) {}
};

class Neuron
{
public:
    Neuron(unsigned, unsigned);
    std::vector<Axon> getAxons();
    void setOutput(double);
    double getOutput();
    void setWeight(unsigned, double);
    void feedForward(const Layer &);
    void backPropagate(Layer &, double);

private:
    unsigned index;
    std::vector<Axon> vecAxons;
    double output;
    double outputSum;

    double randomWeight();
    double sigmoid(double);
    double sigmoidDerivative(double);
};

#endif

#include "neuron.hpp"

Neuron::Neuron(unsigned index, unsigned nAxons)
{
    this->index = index;
    this->output = 0.0;

    
    for (unsigned a = 0; a < nAxons; a++) {
        this->vecAxons.push_back(Axon(randomWeight()));
    }
}

double Neuron::randomWeight()
{
    return rand() / double(RAND_MAX);
}

std::vector<Axon> Neuron::getAxons()
{
    return this->vecAxons;
}

void Neuron::setOutput(double output)
{
    this->output = output;
}

double Neuron::getOutput()
{
    return this->output;
}

void Neuron::setWeight(unsigned axon, double weight)
{
    this->vecAxons[axon].weight = weight;
}

void Neuron::feedForward(const Layer &vecPreviousLayer)
{
    this->outputSum = 0.0;

   
    for (unsigned n = 0; n < vecPreviousLayer.size(); n++) {
        this->outputSum += vecPreviousLayer[n].output * vecPreviousLayer[n].vecAxons[this->index].weight;
    }

    
    this->output = sigmoid(this->outputSum);
}

