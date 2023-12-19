#include "NeuralNetworkSubsystem.h"

#include "utility"

void NeuralNetworkSubsystem::InitNeuralNetwork(const ActivationType inActivation,
												const int inputLayerSize, std::vector<double> inputLayerBiases,
												std::vector<std::vector<double>> inputLayerWeights,
												int hiddenLayers, int hiddenLayersSizes,
												std::vector<std::vector<double>> hiddenLayerBiases,
												std::vector<std::vector<std::vector<double>>> hiddenLayerWeights,
												int outputLayerSize, std::vector<double> outputLayerBiases,
												std::vector<std::vector<double>> outputLayerWeights)
{
	CurrentNeuralNetwork = NeuralNetwork();

	// Reserve the input, output and hidden layers
	CurrentNeuralNetwork.layers.reserve(hiddenLayers + 2);

	Layer inputLayer;
	inputLayer.activation = inActivation;
	inputLayer.numNeurons = inputLayerSize;
	inputLayer.biases = std::move(inputLayerBiases);
	inputLayer.weights = std::move(inputLayerWeights);

	CurrentNeuralNetwork.AddLayer(inputLayer);

	for (int i = 0; i < hiddenLayers; i++)
	{
		Layer hiddenLayer;
		hiddenLayer.activation = inActivation;
		hiddenLayer.numNeurons = hiddenLayersSizes;
		hiddenLayer.biases = hiddenLayerBiases[i];
		hiddenLayer.weights = hiddenLayerWeights[i];

		CurrentNeuralNetwork.AddLayer(hiddenLayer);
	}

	Layer outputLayer;
	outputLayer.activation = inActivation;
	outputLayer.numNeurons = outputLayerSize;
	outputLayer.biases = std::move(outputLayerBiases);
	outputLayer.weights = std::move(outputLayerWeights);
	
	CurrentNeuralNetwork.AddLayer(outputLayer);
}

NeuralNetwork& NeuralNetworkSubsystem::GetNeuralNetwork()
{
	return CurrentNeuralNetwork;
}
