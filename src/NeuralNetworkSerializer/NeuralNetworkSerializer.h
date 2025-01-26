#pragma once
#include <string>

#include "core/HyperParameters.h"
#include "core/NeuralNetwork.h"
#include "subsystems/NeuralNetworkSubsystem.h"

class NeuralNetworkSerializer
{
public:
    static bool SaveToJSON(const std::string& filePath,
                           const NeuralNetwork& network,
                           const HyperParameters& hyperParams,
                           const NeuralNetworkSubsystem::TrainingTimer& trainingTimer,
                           int currentEpoch,
                           int totalEpochs,
                           const std::vector<NeuralNetworkSubsystem::TrainingMetricPoint>& trainingHistory);

    static bool LoadFromJSON(const std::string& filePath,
                             NeuralNetwork& outNetwork,
                             HyperParameters& outHyperParams,
                             NeuralNetworkSubsystem::TrainingTimer& outTimer,
                             int& outCurrentEpoch,
                             int& outTotalEpochs,
                             std::vector<NeuralNetworkSubsystem::TrainingMetricPoint>& outTrainingHistory);
};
