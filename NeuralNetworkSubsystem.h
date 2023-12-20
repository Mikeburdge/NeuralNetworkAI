#pragma once
#include "Cost.h"
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

	void InitNeuralNetwork(const ActivationType& inActivation, const CostType& inCost, const int inputLayerSize,
	                       int
	                       hiddenLayers,
	                       int hiddenLayersSizes,
	                       int outputLayerSize);

	NeuralNetwork& GetNeuralNetwork();
	void StartNeuralNetwork(const std::vector<double>& inputData);
};
