
#include "NeuralNetwork.h"

#include "iostream"
#include "utility"
#include "vector"


NeuralNetwork::NeuralNetwork() = default;

void NeuralNetwork::AddLayer(const Layer& layer)
{
	layers.push_back(layer); // Add the provided layer to the vector of layers.
}

std::vector<double> NeuralNetwork::ForwardPropagation(std::vector<double> inputData) const
{
	std::vector<double> currentInput = std::move(inputData);

	// Iterate through layers
	for (const Layer& layer : layers)
	{
		currentInput = layer.computeOutput(currentInput); // Compute output of the current layer
		// Optionally, print or visualize layer outputs here for better understanding
	}

	return currentInput; // Return the final output of the neural network
}


void NeuralNetwork::BackwardPropagation(std::vector<double> expectedOutput)
{
}
