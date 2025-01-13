#include "NeuralNetworkSubsystem.h"

#include <iostream>
#include <utility>
#include <vulkan/vulkan_core.h>
#include <core/HyperParameters.h>
#include <logging/Logger.h>


void NeuralNetworkSubsystem::InitNeuralNetwork(const ActivationType& inActivation, const CostType& inCost,
                                               const int inputLayerSize,
                                               const int hiddenLayers, const int hiddenLayersSizes,
                                               const int outputLayerSize)
{
    CurrentNeuralNetwork = NeuralNetwork();

    HyperParameters::cost = inCost;
    HyperParameters::activationType = inActivation;

    // Reserve the input, output and hidden layers
    CurrentNeuralNetwork.layers.reserve(hiddenLayers + 2);


    // For now as every hidden layer has the same number of neurons we can just pass the hiddenLayersSizes in as the out neurons
    const Layer inputLayer(inActivation, inCost, inputLayerSize, 0);

    CurrentNeuralNetwork.AddLayer(inputLayer);


    for (int i = 0; i < hiddenLayers; i++)
    {
        int neuronsOut;
        if (i == 0)
        {
            neuronsOut = inputLayerSize; // Nodes coming out of the last layer
        }
        else
        {
            neuronsOut = hiddenLayersSizes;
        }
        const Layer hiddenLayer(inActivation, inCost, hiddenLayersSizes, neuronsOut);

        CurrentNeuralNetwork.AddLayer(hiddenLayer);
    }

    const Layer outputLayer(inActivation, inCost, outputLayerSize, hiddenLayersSizes);

    CurrentNeuralNetwork.AddLayer(outputLayer);
}

NeuralNetwork& NeuralNetworkSubsystem::GetNeuralNetwork()
{
    return CurrentNeuralNetwork;
}

void NeuralNetworkSubsystem::StartNeuralNetwork(const std::vector<double>& inputData,
                                                const std::vector<double>& targetOutput)
{
    PROFILE_LOG;
    NeuralNetwork& network = GetNeuralNetwork();

    const CostType currentCost = HyperParameters::cost;

    for (int epoch = 0; epoch < HyperParameters::epochs; ++epoch)
    {
        // Perform forward propagation
        std::vector<double> predictions = network.ForwardPropagation(inputData);

        // Calculate Cost and Log Progress
        double cost = Cost::CalculateCost(currentCost, predictions, targetOutput);

        std::cout << "Epoch " << epoch << " - Cost: " << cost << '\n';

        // Calculate Error Gradient for Backpropogation
        std::vector<double> costGradient = Cost::CalculateCostDerivative(currentCost, predictions, targetOutput);

        // Perform Backwards Propagation
        network.BackwardPropagation(costGradient);

        // if (epoch % ::HyperParameters::visualizationUpdateInterval == 0 &&  visualizationCallback)
        {
            visualizationCallback(CurrentNeuralNetwork);
        }
    }

    // const int outputLayerIndex = network.layers.size() - 1;
    //
    // Layer outputLayer = CurrentNeuralNetwork.layers[outputLayerIndex];
    //Cost::CalculateCost HERE LAST

    // Example: Print or process the output

    // Further actions after the propagation can be added here
}

void NeuralNetworkSubsystem::SetVisualizationCallback(std::function<void(const NeuralNetwork&)> callback)
{
    visualizationCallback = std::move(callback);
}