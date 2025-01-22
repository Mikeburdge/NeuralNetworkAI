#pragma once
#include <string>

#include "core/HyperParameters.h"
#include "core/NeuralNetwork.h"

class NeuralNetworkSerializer
{
public:


    static bool SaveToJSON(const std::string filePath, const NeuralNetwork& neuralNetwork, const HyperParameters& hyperParameters, int currentEpoch, int totalEpochs);
};
