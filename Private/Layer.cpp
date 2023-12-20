#include "Layer.h"

#include "Activation.h"
#include "HyperParameters.h"
#include "vector"


std::vector<double> Layer:: computeOutput(const std::vector<double>& input)
{
	// Perform matrix multiplication of weights with the input
	std::vector<double> output(numNeurons, 0.0);
	for (int i = 0; i < numNeurons; ++i) {
		for (int j = 0; j < input.size(); ++j) {
			output[i] += weights[i][j] * input[j];
		}
		// Add bias and apply activation function (for example, sigmoid)
		output[i] += biases[i];

		
		neurons[i].ActivationValue = output[i] = Activation::CalculateActivation(activation, output[i]); // Using sigmoid activation here, you can switch to other functions.
		
	}
	return output;
}

void Layer::adjustWeights(const std::vector<double>& errorGradient)
{
	const double learningRate = HyperParameters::learningRate;

	// Update weights based on the error gradient
	for (int i = 0; i < numNeurons; ++i) {
		for (int j = 0; j < weights[i].size(); ++j) {
			weights[i][j] -= learningRate * errorGradient[i] * Activation::CalculateActivation(activation, weights[i][j]); // Update weights using the sigmoid derivative
		}
		biases[i] -= learningRate * errorGradient[i]; // Update biases
	}
}
