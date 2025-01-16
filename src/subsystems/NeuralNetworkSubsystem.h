#pragma once

#include <filesystem>
#include <functional>
#include <vector>
#include "core/SingletonBase.h"
#include "core/NeuralNetwork.h"
#include "dataloader/MNISTDataSet.h"


static const std::string DEFAULT_IMAGES_PATH = (std::filesystem::current_path() / "TrainingData" / "Archive"/ "train-images.idx3-ubyte").string();
static const std::string DEFAULT_LABELS_PATH = (std::filesystem::current_path() / "TrainingData" / "Archive"/ "train-labels.idx1-ubyte").string();

class NeuralNetworkSubsystem : public SingletonBase
{
public:
    static NeuralNetworkSubsystem& GetInstance()
    {
        static NeuralNetworkSubsystem instance;
        return instance;
    }
    NeuralNetworkSubsystem() = default;

    // We do not allow copying
    void operator=(const NeuralNetworkSubsystem&) = delete;

private:
    NeuralNetwork CurrentNeuralNetwork;

    // The MNIST dataset loaded in memory
    MNISTDataSet trainingDataSet;
    bool mnistTrainingDataLoaded = false;

public:
    void InitNeuralNetwork(const ActivationType& inActivation, const CostType& inCost,
                           int inputLayerSize, int hiddenLayers, int hiddenLayerSize,
                           int outputLayerSize);

    NeuralNetwork& GetNeuralNetwork();

    // --------------------------------------------------------------------
    // The old single-sample method (might still come in handy)
    void StartNeuralNetwork(const std::vector<double>& inputData,
                            const std::vector<double>& targetOutput);

    // A function to load MNIST training data from disk
    bool LoadMNISTTrainingData(const std::string& imagesPath,
                               const std::string& labelsPath);

    bool IsMNISTTrainingDataLoaded() const { return mnistTrainingDataLoaded; }
    MNISTDataSet& GetTrainingDataSet() { return trainingDataSet; }

    // --------------------------------------------------------------------
    // Newer multi-batch training with the entire dataset
    // This replaces or extends the "StartNeuralNetwork" single sample approach method.
    void TrainOnMNIST();
    void TrainOnMNISTFullProcess();

    // Visualization callback for real-time network updates, if desired
    void SetVisualizationCallback(std::function<void(const NeuralNetwork&)> callback);

private:
    std::function<void(const NeuralNetwork&)> visualizationCallback;
};
