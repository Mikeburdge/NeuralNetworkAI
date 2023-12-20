#pragma once

#include <vector>
#include "Layer.h"

class NeuralNetwork {


public:
	
	// Vector to store layers in the neural network.
	std::vector<Layer> layers;

	NeuralNetwork();

	/// <summary>
	/// Function to add a new layer to the network.
	/// </summary>
	/// <param name="layer">The Layer object to be added to the network.</param>
	void AddLayer(const Layer& layer);

	/// <summary>
	/// Function to pass input data through the network.
	/// </summary>
	/// <param name="inputData">Input data to be propagated through the network.</param>
	/// <returns>Output data after forward propagation.</returns>
	std::vector<double> ForwardPropagation(std::vector<double> inputData);

	/// <summary>
	/// Function to adjust weights based on errors during training.
	/// </summary>
	/// <param name="expectedOutput">The expected output from the network.</param>
	void BackwardPropagation(std::vector<double> expectedOutput);

};


