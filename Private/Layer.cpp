#include "Layer.h"

#include <random>

#include "Activation.h"
#include "HyperParameters.h"
#include "vector"
#pragma optimize("", off)

std::vector<double> Layer::computeOutput(const std::vector<double>& input)
{
	std::vector<double> output(numNeurons, 0.0);

	for (int neuronsIn = 0; neuronsIn < numNeurons; ++neuronsIn) {
		double neuronOutput = biases[neuronsIn]; // Start with the bias

		for (int neuronsOut = 0; neuronsOut < numNeuronsOutOfPreviousLayer; ++neuronsOut) {
			neuronOutput += input[neuronsOut] * weights[neuronsIn][neuronsOut]; // Sum the weighted inputs
		}

		// Add the neuron's output to the layer's output
		neurons[neuronsIn].ActivationValue = output[neuronsIn] = Activation::CalculateActivation(activation, neuronOutput);
	}

	return output;
}

void Layer::adjustWeights(const std::vector<double>& errorGradient)
{
	const double learningRate = HyperParameters::learningRate;

	// Update weights based on the error gradient
	for (int i = 0; i < numNeurons; ++i) {
		for (int j = 0; j < weights[i].size(); ++j)
		{
			weights[i][j] -= learningRate * errorGradient[i] * Activation::CalculateActivation(activation, weights[i][j]);
		}
		biases[i] -= learningRate * errorGradient[i]; // Update biases
	}
}

void Layer::InitializeRandomBiases(std::mt19937& rng)
{	std::normal_distribution<double> distribution(0.0, 1.0);

	for (double& bias : biases)
	{
		const double randValue = distribution(rng) / sqrt(biases.size());
		bias = randValue;
	}
}

void Layer::InitializeRandomWeights(std::mt19937& rng)
{
	std::normal_distribution<double> distribution(0.0, 1.0);

	for (int i = 0; i < numNeurons; ++i) {
		for (int j = 0; j < numNeuronsOutOfPreviousLayer; ++j) {
			const double randValue = distribution(rng) / sqrt(numNeuronsOutOfPreviousLayer);
			weights[i][j] = randValue;
		}
	}
}



#pragma optimize("", on)