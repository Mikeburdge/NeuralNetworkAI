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

std::vector<double> NeuralNetwork::ForwardPropagation(std::vector<double> inputData, const bool bIsTraining)
{
    PROFILE_LOG;
    storedInput = inputData;

    std::vector<double> currentInput = std::move(inputData);

    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++)
    {
        // Iterate through layers
        currentInput = layers[layerIndex].computeOutput(currentInput, bIsTraining); // Compute output of the current layer
    }

    return currentInput; // Return the final output of the neural network
}


void NeuralNetwork::BackwardPropagation(const std::vector<double>& costGradient)
{
    PROFILE_LOG;
    std::vector<double> errorGradient = costGradient;

    for (int layerIndex = static_cast<int>(layers.size()) - 1; layerIndex >= 0; --layerIndex)
    {
        std::vector<double> prevActivations;

        if (layerIndex == 0)
        {
            // Use stored raw input for the first layer
            prevActivations = storedInput;
        }
        else
        {
            // Get activations from the previous layer
            const Layer& prevLayer = layers[layerIndex - 1];
            prevActivations.resize(prevLayer.numNeurons);
            for (int i = 0; i < prevLayer.numNeurons; ++i)
            {
                prevActivations[i] = prevLayer.neurons[i].ActivationValue;
            }
        }

        // Update weights of the current layer
        layers[layerIndex].adjustWeights(errorGradient, prevActivations);

        // Calculate error gradient for the previous layer
        if (layerIndex > 0)
        {
            const Layer& currentLayer = layers[layerIndex];
            const Layer& prevLayer = layers[layerIndex - 1];
            errorGradient = currentLayer.CalculatePreviousLayerError(errorGradient, prevLayer.neurons);
        }
    }
}