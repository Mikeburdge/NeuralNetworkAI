#pragma once
#include <vector>

#include "Activation.h"
#include "Neuron.h"


class Layer {
	
public:

	ActivationType activation; // Activation type we are using.

	int numNeurons; // Number of neurons in the layer.
	std::vector<Neuron> neurons; // Neurons.

	std::vector<std::vector<double>> weights; // Weights matrix.
	std::vector<double> biases; // Biases vector.

	Layer(const ActivationType inActivation, const int inNum_neurons, const std::vector<std::vector<double>>& inWeights, const std::vector<double>& inBiases)
		: activation(inActivation),
		  numNeurons(inNum_neurons),
		  weights(inWeights),
		  biases(inBiases)
	{
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

};


