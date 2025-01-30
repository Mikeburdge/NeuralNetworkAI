#pragma once

#include <atomic>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "core/NeuralNetwork.h"
#include "core/SingletonBase.h"
#include "dataloader/MNISTDataSet.h"


static const std::string DEFAULT_TRAIN_IMAGES_PATH = (std::filesystem::current_path() / "TrainingData" / "Archive" / "train-images.idx3-ubyte").string();
static const std::string DEFAULT_TRAIN_LABELS_PATH = (std::filesystem::current_path() / "TrainingData" / "Archive" / "train-labels.idx1-ubyte").string();
static const std::string DEFAULT_TEST_IMAGES_PATH = (std::filesystem::current_path() / "TrainingData" / "Archive" / "t10k-images.idx3-ubyte").string();
static const std::string DEFAULT_TEST_LABELS_PATH = (std::filesystem::current_path() / "TrainingData" / "Archive" / "t10k-labels.idx1-ubyte").string();

class NeuralNetworkSubsystem : public SingletonBase
{
public:
    struct TrainingTimer
    {
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastEpochTime;
        double epochDuration = 0.0;
        bool isInitialized = false;
    };

    static TrainingTimer trainingTimer;

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
    bool bIsNeuralNetworkInitialized = false;

    // The MNIST dataset loaded in memory
    MNISTDataSet trainingDataSet;
    bool bIsMnistTrainingDataLoaded = false;

    MNISTDataSet testDataSet;
    bool bIsMnistTestDataLoaded = false;

    int vizUpdateInterval = 10;
    int vizBatchCounter = 0;

public:
    // Threadding
    std::thread trainingThread;
    std::atomic<bool> trainingInProgressAtomic{false};

    std::atomic<int> currentEpochAtomic{0};
    std::atomic<double> currentLossAtomic{0.0};
    std::atomic<double> currentAccuracyAtomic{0.0};
    std::atomic<int> totalEpochsAtomic{0};

    std::atomic<int> correctPredictionsThisBatchAtomic{0};
    std::atomic<int> currentBatchSizeAtomic{0};
    std::atomic<int> totalCorrectPredictionsAtomic{0};
    std::atomic<int> totalPredictionsAtomic{0};

    std::deque<bool> pastPredictionResults;
    std::atomic<double> rollingAccuracyAtomic{0.0};

    int pastPredictionCorrectCount = 0;

    std::atomic<int> currentBatchIndexAtomic{0};
    std::atomic<int> totalBatchesInEpochAtomic{0};

    // for stopping the training
    std::atomic<bool> stopRequestedAtomic{false};

    // more metrics
    std::atomic<double> totalBatchTimeAtomic{0.0};
    std::atomic<double> averageBatchTimeAtomic{0.0};
    std::atomic<double> samplesPerSecAtomic{0.0};

    // Getter functions for UI to access test set results
    const std::vector<std::vector<float>>& GetTestSetImages() const { return testSetImages; }
    const std::vector<int>& GetTestSetPredictions() const { return testSetPredictions; }
    const std::vector<double>& GetTestSetConfidence() const { return testSetConfidence; }

    // Graph Stuff
    struct TrainingMetricPoint
    {
        float timeSeconds; // seconds since training started
        float loss; // Y-axis
        float accuracy; // Y-axis 
        float rollingAcc; // Y-axis
    };


    std::vector<TrainingMetricPoint> trainingHistory;
    std::mutex metricMutex;
    // Emable / disable
    std::atomic<bool> showLossGraph{true};
    std::atomic<bool> showAccuracyGraph{true};
    std::atomic<bool> showRollingAccuracyGraph{true};


    void SetVizUpdateInterval(int interval)
    {
        vizUpdateInterval = interval;
    }

    // number of neurons to display in one layer
    int maxNeuronsToDisplay = 20;

    void InitNeuralNetwork(const ActivationType& inActivation, const CostType& inCost,
                           int inputLayerSize, int hiddenLayers, int hiddenLayerSize,
                           int outputLayerSize);

    NeuralNetwork& GetNeuralNetwork();

    bool LoadMNISTTrainingData(const std::string& imagesPath,
                               const std::string& labelsPath);

    bool IsMNISTTrainingDataLoaded() const { return bIsMnistTrainingDataLoaded; }
    MNISTDataSet& GetTrainingDataSet() { return trainingDataSet; }

    bool LoadMNISTTestData(const std::string& imagesPath,
                           const std::string& labelsPath);

    double EvaluateTestSet();

    bool IsMNISTTestDataLoaded() const { return bIsMnistTestDataLoaded; }
    MNISTDataSet& GetTestDataSet() { return testDataSet; }

    // --------------------------------------------------------------------
    // New batch training with the entire dataset
    // This extends on the "StartNeuralNetwork" single sample approach method.
    void TrainOnMNIST();
    void TrainOnMNISTFullProcess();

    void TrainOnMNISTAsync(); // This will call the TrainOnMNIST functions
    bool IsTrainingInProgress() const { return trainingInProgressAtomic.load(); }

    void StopTraining();


    // Saving and Loading
    bool SaveNetwork(const std::string& filePath);
    bool LoadNetwork(const std::string& filePath);
    std::pair<int, double> InferSingleImageFromPath(const std::string& path);


    // To call whe ninferring an image
    std::pair<int, double> InferSingleImage(const std::vector<double>& image);
    std::vector<double> LoadAndProcessPNG(const std::string& path);


    // functions for stopRequested atomic variable
    void RequestStopTraining() { stopRequestedAtomic.store(true); }
    bool IsStopRequested() const { return stopRequestedAtomic.load(); }


    // Visualization callback for real-time network updates, if desired
    void SetVisualizationCallback(std::function<void(const NeuralNetwork&)> callback);
    void TestCustomSet();

private:
    std::function<void(const NeuralNetwork&)> visualizationCallback;


    void TrainOnMNISTThreadEntry();

    static double SumDoubles(const std::vector<double>& values);

private:
    // Storage for custom test set results
    std::vector<std::vector<float>> testSetImages;
    std::vector<int> testSetPredictions;
    std::vector<double> testSetConfidence;
};
