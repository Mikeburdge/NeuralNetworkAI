#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <thread>
#include <vector>

#include "core/NeuralNetwork.h"
#include "core/SingletonBase.h"
#include "dataloader/MNISTDataSet.h"


static const std::string DEFAULT_IMAGES_PATH = (std::filesystem::current_path() / "TrainingData" / "Archive" / "train-images.idx3-ubyte").string();
static const std::string DEFAULT_LABELS_PATH = (std::filesystem::current_path() / "TrainingData" / "Archive" / "train-labels.idx1-ubyte").string();

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
    bool bIsMnistTrainingDataLoaded = false;

    // Threadding
    std::thread trainingThread;
    std::atomic<bool> trainingInProgress{false};

    // for stopping the training
    std::atomic<bool> stopRequested{false};


public:

    // number of neurons to display in one layer
    int maxNeuronsToDisplay = 30;
    
    void InitNeuralNetwork(const ActivationType& inActivation, const CostType& inCost,
                           int inputLayerSize, int hiddenLayers, int hiddenLayerSize,
                           int outputLayerSize);

    NeuralNetwork& GetNeuralNetwork();

    // --------------------------------------------------------------------
    // The old single sample method (might still come in handy)
    void StartNeuralNetwork(const std::vector<double>& inputData,
                            const std::vector<double>& targetOutput);

    // A function to load MNIST training data from disk
    bool LoadMNISTTrainingData(const std::string& imagesPath,
                               const std::string& labelsPath);

    bool IsMNISTTrainingDataLoaded() const { return bIsMnistTrainingDataLoaded; }
    MNISTDataSet& GetTrainingDataSet() { return trainingDataSet; }

    // --------------------------------------------------------------------
    // New batch training with the entire dataset
    // This extends on the "StartNeuralNetwork" single sample approach method.
    void TrainOnMNIST();
    void TrainOnMNISTFullProcess();

    void TrainOnMNISTAsync(); // This will call the TrainOnMNIST functions
    bool IsTrainingInProgress() const { return trainingInProgress.load(); }

    void StopTraining();


    // Saving and Loading
    bool SaveNetwork(const std::string& filePath);
    bool LoadNetwork(const std::string& filePath);


    // To call whe ninferring an image
    int InferSingleImage(const std::vector<double>& image);


    // functions for stopRequested atomic variable
    void RequestStopTraining() { stopRequested.store(true); }
    bool IsStopRequested() const { return stopRequested.load(); }


    // Visualization callback for real-time network updates, if desired
    void SetVisualizationCallback(std::function<void(const NeuralNetwork&)> callback);

private:
    std::function<void(const NeuralNetwork&)> visualizationCallback;


    void TrainOnMNISTThreadEntry();
    
};
