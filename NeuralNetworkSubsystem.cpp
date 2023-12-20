#include "NeuralNetworkSubsystem.h"

void NeuralNetworkSubsystem::InitNeuralNetwork(const ActivationType inActivation,
                                               const int inputLayerSize, const std::vector<double>& inputLayerBiases,
                                               const std::vector<std::vector<double>>& inputLayerWeights,
                                               int hiddenLayers, int hiddenLayersSizes,
                                               std::vector<std::vector<double>> hiddenLayerBiases,
                                               std::vector<std::vector<std::vector<double>>> hiddenLayerWeights,
                                               int outputLayerSize, const std::vector<double>& outputLayerBiases,
                                               const std::vector<std::vector<double>>& outputLayerWeights)
{
	CurrentNeuralNetwork = NeuralNetwork();

	// Reserve the input, output and hidden layers
	CurrentNeuralNetwork.layers.reserve(hiddenLayers + 2);

	Layer inputLayer(inActivation, inputLayerSize, inputLayerWeights, inputLayerBiases);

	CurrentNeuralNetwork.AddLayer(inputLayer);

	for (int i = 0; i < hiddenLayers; i++)
	{
		Layer hiddenLayer(inActivation, hiddenLayersSizes, hiddenLayerWeights[i], hiddenLayerBiases[i]);

		CurrentNeuralNetwork.AddLayer(hiddenLayer);
	}

	Layer outputLayer(inActivation, outputLayerSize, outputLayerWeights, outputLayerBiases);
	
	CurrentNeuralNetwork.AddLayer(outputLayer);
}

NeuralNetwork& NeuralNetworkSubsystem::GetNeuralNetwork()
{
	return CurrentNeuralNetwork;
}
