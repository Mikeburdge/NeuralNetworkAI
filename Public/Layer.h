#pragma once
#include <random>
#include <vector>

#include "Activation.h"
#include "Cost.h"
#include "Neuron.h"

class Layer
{
public:
    ActivationType activation; // Activation type we are using.

    CostType cost;

    int numNeurons; // Number of neurons in the layer.
    int numNeuronsOutOfPreviousLayer;

    std::vector<Neuron> neurons; // Neurons.

    std::vector<std::vector<double>> weights; // Weights matrix.
    std::vector<double> biases; // Biases vector.
    
    bool isInputLayer = false;

    Layer(const ActivationType& inActivation, const CostType& inCost, const int inNumNeurons, const int inNumNeuronsOut)
        : activation(inActivation), cost(inCost),
          numNeurons(inNumNeurons), numNeuronsOutOfPreviousLayer(inNumNeuronsOut),
          isInputLayer(inNumNeuronsOut == 0)
    {
        // Initialize weights only if this is not the first layer (input layer)
        if (!isInputLayer)
        {
            weights = std::vector(numNeurons, std::vector<double>(numNeuronsOutOfPreviousLayer));
            biases = std::vector<double>(numNeurons);
        }
        
        std::random_device rd;
        std::mt19937 rng(rd());

        InitializeRandomBiases(rng);
        InitializeRandomWeights(rng);

        neurons.reserve(numNeurons);
        for (int i = 0; i < numNeurons; i++)
        {
            neurons.emplace_back(1.0);
        }
    }

    /// <summary>
    /// Function to calculate the output of the layer given input.
    /// </summary>
    /// <param name="input">Input data to the layer.</param>
    /// <returns>Output data from the layer.</returns>
    std::vector<double> computeOutput(const std::vector<double>& input);

    /// <summary>
    /// Function to update weights during backpropagation.
    /// </summary>
    /// <param name="errorGradient">The gradient of the error with respect to the output.</param>
    void adjustWeights(const std::vector<double>& errorGradient);

    void InitializeRandomBiases(std::mt19937& rng);
    void InitializeRandomWeights(std::mt19937& rng);
    std::vector<double> CalculatePreviousLayerError(const std::vector<double>& currentLayersErrorGradient,
                                                    const std::vector<Neuron>& previousLayerNeurons) const;
};
