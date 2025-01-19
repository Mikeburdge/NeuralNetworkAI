#include "NeuralNetwork.h"

#include "iostream"
#include "utility"
#include "vector"
#include "../Logging/Logger.h"


NeuralNetwork::NeuralNetwork() = default;

void NeuralNetwork::AddLayer(const Layer& layer)
{
    PROFILE_LOG;
    layers.push_back(layer); // Add the provided layer to the vector of layers.
}

std::vector<double> NeuralNetwork::ForwardPropagation(std::vector<double> inputData)
{
    PROFILE_LOG;
    std::vector<double> currentInput = std::move(inputData);

    // Iterate through layers
    for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++)
    {
        currentInput = layers[layerIndex].computeOutput(currentInput); // Compute output of the current layer
    }

    return currentInput; // Return the final output of the neural network
}


void NeuralNetwork::BackwardPropagation(const std::vector<double>& costGradient)
{
    PROFILE_LOG;
    std::vector<double> errorGradient = costGradient;

    for (int layerIndex = layers.size() - 1; layerIndex >= 0; --layerIndex)
    {
        Layer& currentLayer = layers[layerIndex];
        if (layerIndex > 0)
        {
            Layer& prevLayer = layers[layerIndex - 1];
            std::vector<double> prevActivations(prevLayer.numNeurons);
            for (int prevLayerIndex = 0; prevLayerIndex < prevLayer.numNeurons; prevLayerIndex++)
            {
                prevActivations[prevLayerIndex] = prevLayer.neurons[prevLayerIndex].ActivationValue;
            }
            currentLayer.adjustWeights(errorGradient, prevActivations);
        }
        else
        {
            currentLayer.adjustWeights(errorGradient, {});
        }

        if (layerIndex > 0)
        {
            Layer& prevLayer = layers[layerIndex - 1];
            errorGradient = currentLayer.CalculatePreviousLayerError(errorGradient, prevLayer.neurons);
        }
    }
}
