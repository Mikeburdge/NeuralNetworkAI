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

    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++)
    {
        // Iterate through layers
        currentInput = layers[layerIndex].computeOutput(currentInput); // Compute output of the current layer
    }

    return currentInput; // Return the final output of the neural network
}


void NeuralNetwork::BackwardPropagation(const std::vector<double>& costGradient)
{
    // Could do this in a more advanced way I think? if we run multiple layers' backpropagation
    // simultaneously it would speed it up more but I'd have to figure out how to deal
    // with the fact each layer needs the result from the previous layer. 
    PROFILE_LOG;
    std::vector<std::thread> layerThreads;
    layerThreads.reserve(layers.size());

    std::vector<double> errorGradient = costGradient;

    for (int layerIndex = layers.size() - 1; layerIndex >= 0; --layerIndex)
    {
        // here i will put the error gradient in a temp layer,
        // and combine them after in the main thread ready for the next iteration

        layerThreads.emplace_back([this, layerIndex, &errorGradient]()
            {
                Layer& currentLayer = layers[layerIndex];
                if (layerIndex > 0)
                {
                    const Layer& prevLayer = layers[layerIndex - 1];
                    std::vector<double> prevActivations(prevLayer.numNeurons);
                    for (int prevLayerIndex = 0; prevLayerIndex < prevLayer.numNeurons; prevLayerIndex++)
                    {
                        prevActivations[prevLayerIndex] = prevLayer.neurons[prevLayerIndex].ActivationValue;
                    }
                    currentLayer.adjustWeights(errorGradient, prevActivations);
                }
            }
        );

        // join/ wait to join the first layer added to the vector
        layerThreads.back().join();
        if (layerIndex > 0)
        {
            const Layer& currentLayer = layers[layerIndex];
            const Layer& prevLayer = layers[layerIndex - 1];
            errorGradient = currentLayer.CalculatePreviousLayerError(errorGradient, prevLayer.neurons);
        }
    }

    // run through all threads
    layerThreads.clear();
}
