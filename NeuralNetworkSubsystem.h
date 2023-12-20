#pragma once
#include "NeuralNetwork.h"
#include "SingletonBase.h"


class NeuralNetworkSubsystem : public SingletonBase
{
public:

	static NeuralNetworkSubsystem& GetInstance() {
		static NeuralNetworkSubsystem instance;
		return instance;
	}

	NeuralNetworkSubsystem() = default;

	// Prevent copying and assignment
	//NeuralNetworkSubsystem(const NeuralNetworkSubsystem&) = delete;
	void operator=(const NeuralNetworkSubsystem&) = delete;

private:

	NeuralNetwork CurrentNeuralNetwork;

public:

	void InitNeuralNetwork(ActivationType inActivation, int inputLayerSize, const std::vector<double>& inputLayerBiases,
	                       const std::vector<std::vector<double>>& inputLayerWeights, int hiddenLayers, int hiddenLayersSizes,
	                       std::vector<std::vector<double>> hiddenLayerBiases,
	                       std::vector<std::vector<std::vector<double>>> hiddenLayerWeights, int outputLayerSize,
	                       const std::vector<double>& outputLayerBiases, const std::vector<std::vector<double>>&
	                       outputLayerWeights);

	NeuralNetwork& GetNeuralNetwork();
};
