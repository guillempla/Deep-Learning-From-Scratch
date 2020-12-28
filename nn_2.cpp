#include "neural-net.hpp"

NeuralNetwork(const std::vector<unsigned> &vecTopology, bool Bias)
{
    this->Bias = Bias;

    
    for (unsigned l = 0; l < vecTopology.size(); l++) {
        this->vecLayers.push_back(Layer());

        unsigned nAxons = (l == vecTopology.size() - 1) ? 0 : vecTopology[l + 1];
        for (unsigned n = 0; n < vecTopology[l] + (this->useBias) ? 1 : 0; n++) {
            this->vecLayers[l].push_back(Neuron(n, nAxons));
        }

        
        if (this->useBias && l == vecTopology.size() - 1)
            this->vecLayers[l].pop_back();
    }
}

void NeuralNetwork::status()
{
    for (unsigned l = 0; l < this->vecLayers.size(); l++) {
        std::cout << "\nLayer " << l;

        Layer &vecLayer = this->vecLayers[l];
        for (unsigned n = 0; n < vecLayer.size(); n++) {
            std::cout << "\n    Neuron " << n << "\n";

            std::vector<Axon> vecAxons = vecLayer[n].getAxons();
            for (unsigned w = 0; w < vecAxons.size(); w++) {
                std::cout << "        Axon " << w << " weight: " << vecAxons[w].weight << "\n";
                std::cout << "        Axon " << w << " output: " << vecLayer[n].getOutput() << "\n\n";
            }

            if (l == this->vecLayers.size() - 1)
                std::cout << "        Output: " << vecLayer[n].getOutput() << "\n\n";
        }
    }
}
void NeuralNetwork::setWeight(unsigned layer, unsigned neuron, unsigned axon, double weight)
{
    this->vecLayers[layer][neuron].setWeight(axon, weight);
}

void NeuralNetwork::feedForward(const std::vector<double> &vecInputs)
{
    
    for (unsigned n = 0; n < vecInputs.size(); n++) {
        this->vecLayers[0][n].setOutput(vecInputs[n]);
    }

    
    for (unsigned l = 1; l < this->vecLayers.size(); l++) {
        Layer &vecLayer = this->vecLayers[l];
        for (unsigned n = 0; n < vecLayer.size(); n++) {
            vecLayer[n].feedForward(this->vecLayers[l - 1]);
        }
    }
}
std::vector<double> NeuralNetwork::getOutput()
{
   
    std::vector<double> vecOutputs;

    Layer &vecLayer = this->vecLayers.back();
    for (unsigned n = 0; n < vecLayer.size(); n++) {
        vecOutputs.push_back(vecLayer[n].getOutput());
    }

    return vecOutputs;
}

