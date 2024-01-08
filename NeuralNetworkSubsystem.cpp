#include "NeuralNetworkSubsystem.h"

#include <utility>

#include "Cost.h"

void NeuralNetworkSubsystem::InitNeuralNetwork(const ActivationType& inActivation, const CostType& inCost,
                                               const int inputLayerSize,
                                               const int hiddenLayers, const int hiddenLayersSizes,
                                               const int outputLayerSize)
{
	CurrentNeuralNetwork = NeuralNetwork();

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

void NeuralNetworkSubsystem::StartNeuralNetwork(const std::vector<double>& inputData) {


	// Perform forward propagation
	std::vector<double> forwardPropagationData = CurrentNeuralNetwork.ForwardPropagation(inputData);

	// Perform back propagation

	const int outputLayerIndex = CurrentNeuralNetwork.layers.size() - 1;

	Layer outputLayer = CurrentNeuralNetwork.layers[outputLayerIndex];

	//Cost::CalculateCost HERE LAST

	// Example: Print or process the output

	// Further actions after the propagation can be added here
}