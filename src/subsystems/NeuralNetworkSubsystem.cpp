#include "NeuralNetworkSubsystem.h"

#include <cstdint>
#include <future>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <core/HyperParameters.h>
#include <logging/Logger.h>
#include <vulkan/vulkan_core.h>

// Include stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "NeuralNetworkSerializer/NeuralNetworkSerializer.h"
#include "utility/NeuralNetworkUtility.h"

NeuralNetworkSubsystem::TrainingTimer NeuralNetworkSubsystem::trainingTimer;

void NeuralNetworkSubsystem::InitNeuralNetwork(const ActivationType& inActivation, const CostType& inCost,
                                               const int inputLayerSize,
                                               const int hiddenLayers, const int hiddenLayerSize,
                                               const int outputLayerSize)
{
    {
        std::lock_guard<std::mutex> lock(metricMutex);
        trainingHistory.clear();
    }

    CurrentNeuralNetwork = NeuralNetwork();

    HyperParameters::cost = inCost;
    HyperParameters::activationType = inActivation;

    ActivationType finalLayerActivation;
    if (inCost == CostType::crossEntropy)
    {
        // Force softmax for crossEntropy
        finalLayerActivation = ActivationType::softmax;
    }
    else
    {
        // Use the user-chosen final-layer activation
        finalLayerActivation = inActivation;
    }


    // Reserve the input, output and hidden layers
    CurrentNeuralNetwork.layers.reserve(hiddenLayers + 1);

    int prevLayerSize = inputLayerSize;

    // build hidden layers
    for (int i = 0; i < hiddenLayers; i++)
    {
        Layer hiddenLayer(inActivation, inCost, hiddenLayerSize, prevLayerSize);
        CurrentNeuralNetwork.AddLayer(hiddenLayer);
        prevLayerSize = hiddenLayerSize;
    }

    // build output layer
    Layer outputLayer(finalLayerActivation, inCost, outputLayerSize, prevLayerSize);
    CurrentNeuralNetwork.AddLayer(outputLayer);

    for (Layer& layer : CurrentNeuralNetwork.layers)
    {
        // make sure we initialize adam so that he can perform at his best :nod:
        layer.InitAdam();
    }
}

NeuralNetwork& NeuralNetworkSubsystem::GetNeuralNetwork()
{
    return CurrentNeuralNetwork;
}


bool NeuralNetworkSubsystem::LoadMNISTTrainingData(const std::string& imagesPath, const std::string& labelsPath)
{
    bool loaded = trainingDataSet.LoadTrainingSet(imagesPath, labelsPath);
    bIsMnistTrainingDataLoaded = loaded;

    if (!loaded)
    {
        LOG(LogLevel::ERROR, "Failed to load MNIST training data. Check file paths!");
    }
    else
    {
        std::string msg = "MNIST training data loaded. Dataset size = " + std::to_string(trainingDataSet.Size());
        LOG(LogLevel::INFO, msg);
    }

    return loaded;
}

bool NeuralNetworkSubsystem::LoadMNISTTestData(const std::string& imagesPath, const std::string& labelsPath)
{
    bool loaded = testDataSet.LoadTrainingSet(imagesPath, labelsPath);
    bIsMnistTestDataLoaded = loaded;

    if (!loaded)
    {
        LOG(LogLevel::ERROR, "Failed to load MNIST test data. Check file paths!");
    }
    else
    {
        LOG(LogLevel::INFO, "MNIST test data loaded. Size = " + std::to_string(testDataSet.Size()));
    }
    return loaded;
}
double NeuralNetworkSubsystem::EvaluateTestSet()
{
    if (!bIsMnistTestDataLoaded)
    {
        bIsMnistTestDataLoaded = LoadMNISTTestData(DEFAULT_TEST_IMAGES_PATH, DEFAULT_TEST_LABELS_PATH);
    }
    
    if (!bIsMnistTestDataLoaded || testDataSet.Size() == 0)
    {
        LOG(LogLevel::ERROR, "No test data loaded. EvaluateTestSet aborted.");
        return 0.0;
    }
    int total = (int)testDataSet.Size();

    unsigned int concurrency = std::thread::hardware_concurrency();
    int numThreads = (concurrency == 0) ? 4 : (int)concurrency;

    std::vector<std::thread> threads;
    std::atomic<int> globalCorrectCountAtomic(0);

    auto worker = [&](int startIdx, int endIdx)
    {
        int localCorrect = 0;
        for (int i = startIdx; i < endIdx; i++)
        {
            const auto& img = testDataSet.GetImages()[i];
            const auto& labelOneHot = testDataSet.GetLabelsOneHot()[i];
            auto out = CurrentNeuralNetwork.ForwardPropagation(img, false);

            int bestIdx = -1;
            double bestVal = -999999.0;
            for (int j = 0; j < (int)out.size(); j++)
            {
                if (out[j] > bestVal)
                {
                    bestVal = out[j];
                    bestIdx = j;
                }
            }
            // find actual
            int actualIdx = -1;
            for (int j = 0; j < (int)labelOneHot.size(); j++)
            {
                if (labelOneHot[j] == 1.0)
                {
                    actualIdx = j;
                    break;
                }
            }
            if (bestIdx == actualIdx)
            {
                localCorrect++;
            }
        }
        globalCorrectCountAtomic.fetch_add(localCorrect);
    };

    int chunkSize = total / numThreads;
    for (int threadIndex = 0; threadIndex < numThreads; threadIndex++)
    {
        int startIdx = threadIndex * chunkSize;
        int endIdx = (threadIndex == numThreads - 1) ? total : startIdx + chunkSize;
        threads.emplace_back(worker, startIdx, endIdx);
    }

    // Join
    for (auto& thread : threads)
    {
        thread.join();
    }

    double accuracy = (double)globalCorrectCountAtomic.load() / (double)total;
    LOG(LogLevel::INFO, "Test set accuracy (multithread) = " + std::to_string(accuracy * 100.0) + "%");
    return accuracy;
}

void NeuralNetworkSubsystem::TrainOnMNIST()
{
    PROFILE_LOG;

    // Neeeds to have data
    if (!bIsMnistTrainingDataLoaded)
    {
        LOG(LogLevel::ERROR, "Failed to load MNIST training data!");
        return;
    }

    if (CurrentNeuralNetwork.layers.empty())
    {
        LOG(LogLevel::ERROR, "Cannot train neural network, no architecture loaded");
        return;
    }

    // being careful now that we are loading epochs
    int epochs = HyperParameters::epochs;
    if (epochs <= 0)
    {
        LOG(LogLevel::ERROR, "No epochs to train. Exiting.");
        return;
    }

    size_t startEpoch = (size_t)currentEpochAtomic.load();
    if (startEpoch >= (size_t)epochs)
    {
        LOG(LogLevel::ERROR, "All epochs completed already.");
        return;
    }

    NeuralNetwork& network = GetNeuralNetwork();
    MNISTDataSet& dataset = GetTrainingDataSet();
    size_t datasetSize = dataset.Size();
    if (datasetSize == 0)
    {
        LOG(LogLevel::ERROR, "Failed to load MNIST training data!");
        return;
    }

    int batchsize = HyperParameters::batchSize;
    CostType currentCost = HyperParameters::cost;

    //  we need an index list so we can shuffle nicely

    std::vector<size_t> indices(datasetSize);
    for (size_t i = 0; i < datasetSize; i++)
    {
        indices[i] = i;
    }

    // Initialize RNG with a fixed seed for reproducibility
    std::mt19937 rng(42);

    // Configure dropout for each layer if enabled
    for (Layer& layer : network.layers)
    {
        if (&layer != &network.layers.back()) // Assuming last layer is output
        {
            // Apply dropout only to hidden layers
            layer.SetDropout(HyperParameters::useDropoutRate, HyperParameters::dropoutRate);
        }
    }

    // Main Training Loop
    totalEpochsAtomic.store(epochs);

    int totalCorrectPredictions = 0;
    totalPredictionsAtomic.store(0);

    for (size_t epoch = startEpoch; epoch < epochs; ++epoch)
    {
        currentEpochAtomic.store(epoch);

        // Shuffle dataset
        std::shuffle(indices.begin(), indices.end(), rng);

        // Calculate total batches for this epoch
        size_t totalBatches = (datasetSize + batchsize - 1) / batchsize; // Ceiling division

        double epochAverageCost = 0.0;
        double epochAccuracy = 0.0;

        currentLossAtomic.store(epochAverageCost);
        currentAccuracyAtomic.store(epochAccuracy);
        totalBatchesInEpochAtomic.store(static_cast<int>(totalBatches));

        // Reset current batch index
        currentBatchIndexAtomic.store(0);

        // Initialize epoch variables
        double epochCostSum = 0.0;
        int numBatches = 0;

        static double accumulatedEpochTime = 0.0;
        static int epochCountForAverage = 0;

        for (size_t startIndex = 0; startIndex < datasetSize; startIndex += batchsize)
        {
            if (stopRequestedAtomic.load())
            {
                LOG(LogLevel::INFO, "Stop mid-epoch...");
                break;
            }

            auto batchStartTime = std::chrono::high_resolution_clock::now();

            size_t endIndex = std::min(startIndex + batchsize, datasetSize); // Again possibly datasetSize - 1 if we hit out of index stuff here
            size_t realBatch = endIndex - startIndex;
            std::vector<std::vector<double>> batchInputs(realBatch);
            std::vector<std::vector<double>> batchTargets(realBatch);

            for (size_t i = 0; i < realBatch; ++i)
            {
                size_t idx = indices[startIndex + i];
                batchInputs[i] = dataset.GetImages()[idx];
                batchTargets[i] = dataset.GetLabelsOneHot()[idx];
            }

            currentBatchSizeAtomic.store((int)realBatch);
            totalPredictionsAtomic.fetch_add((int)realBatch);

            // Parallelize 
            std::vector<std::future<std::vector<double>>> futures;
            futures.reserve(realBatch);

            std::vector<std::vector<double>> batchPredictions(realBatch);

            for (size_t i = 0; i < realBatch; ++i)
            {
                futures.push_back(std::async(std::launch::async, [&network, &batchInputs, i]()
                {
                    return network.ForwardPropagation(batchInputs[i]);
                }));
            }

            std::vector<double> perSampleCosts(realBatch, 0.0);
            int localCorrectCount = 0;

            for (size_t i = 0; i < realBatch; ++i)
            {
                batchPredictions[i] = futures[i].get();
                double cost = Cost::CalculateCost(currentCost, batchPredictions[i], batchTargets[i]);
                perSampleCosts[i] = cost;

                // predicted index vs actual
                int bestPredictionIndex = -1;
                double bestValue = 0.0;
                for (int j = 0; j < (int)batchPredictions[i].size(); j++)
                {
                    if (batchPredictions[i][j] > bestValue)
                    {
                        bestValue = batchPredictions[i][j];
                        bestPredictionIndex = j;
                    }
                }

                int actualIndex = -1;
                for (int j = 0; j < (int)batchTargets[i].size(); j++)
                {
                    if (batchTargets[i][j] == 1)
                    {
                        actualIndex = j;
                        break;
                    }
                }
                if (bestPredictionIndex == actualIndex)
                {
                    localCorrectCount++;
                    totalCorrectPredictions++;

                    pastPredictionResults.push_back(true);
                    pastPredictionCorrectCount++;
                }
                else
                {
                    pastPredictionResults.push_back(false);
                }

                while (pastPredictionResults.size() > 1000)
                {
                    if (pastPredictionResults.front())
                    {
                        pastPredictionCorrectCount--;
                    }
                    pastPredictionResults.pop_front();
                }

                rollingAccuracyAtomic.store(
                    static_cast<double>(pastPredictionCorrectCount) / pastPredictionResults.size()
                );
            }
            double batchCost = SumDoubles(perSampleCosts);

            currentLossAtomic.store(batchCost / (double)realBatch);

            correctPredictionsThisBatchAtomic.store(localCorrectCount);
            totalCorrectPredictionsAtomic.store(totalCorrectPredictions);

            double batchAccuracy = (double)localCorrectCount / (double)realBatch;
            epochAccuracy += batchAccuracy;

            batchCost /= (double)realBatch;
            epochCostSum += batchCost;
            ++numBatches;

            // Cost Gradient stuff for each sample.. now with all new pparrallellizzattionn
            // Theres a chance to do a better, more sophisticated

            std::vector<std::future<std::vector<double>>> gradientFutures;
            gradientFutures.reserve(realBatch);

            for (size_t i = 0; i < realBatch; ++i)
            {
                gradientFutures.push_back(std::async(std::launch::async, [&batchPredictions, &batchTargets, i]()
                {
                    return Cost::CalculateCostDerivative(HyperParameters::cost, batchPredictions[i], batchTargets[i]);
                }));
            }

            std::vector<double> totalGradient(network.layers.back().numNeurons, 0.0);

            for (size_t i = 0; i < realBatch; ++i)
            {
                std::vector<double> gradientPartial = gradientFutures[i].get();
                for (size_t j = 0; j < gradientPartial.size(); ++j)
                {
                    totalGradient[j] += gradientPartial[j];
                }
            }

            // Updates each batch

            auto batchEndTime = std::chrono::high_resolution_clock::now();
            double batchDuration = std::chrono::duration_cast<std::chrono::duration<double>>(batchEndTime - batchStartTime).count();

            totalBatchTimeAtomic.store(batchDuration);

            double oldAvg = averageBatchTimeAtomic.load();
            double newAvg = oldAvg * 0.9 + (batchDuration) * 0.1;

            averageBatchTimeAtomic.store(newAvg);

            double sps = (double)realBatch / batchDuration;
            samplesPerSecAtomic.store(sps);

            // Average the accuracy
            epochAccuracy /= (double)numBatches;

            double partialAccuracy = (double)totalCorrectPredictions / (double)totalPredictionsAtomic.load();
            currentAccuracyAtomic.store(partialAccuracy);

            // Average the gradient
            for (size_t j = 0; j < totalGradient.size(); ++j)
            {
                totalGradient[j] /= (double)realBatch;
            }

            // Use backpropogation on the averaghe gradient
            network.BackwardPropagation(totalGradient);

            // update the visuals

            vizBatchCounter++;
            if (vizBatchCounter % vizUpdateInterval == 0)
            {
                if (visualizationCallback)
                    visualizationCallback(CurrentNeuralNetwork);
            }

            // Increment current batch index
            currentBatchIndexAtomic.fetch_add(1);

            // Graph Stuff
            std::lock_guard<std::mutex> lock(metricMutex);

            double elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - trainingTimer.startTime).count();


            // create a data point
            TrainingMetricPoint dataPoint;
            dataPoint.timeSeconds = static_cast<float>(elapsedSeconds);
            dataPoint.loss = static_cast<float>(batchCost);
            dataPoint.accuracy = static_cast<float>(partialAccuracy * 100.f);
            dataPoint.rollingAcc = static_cast<float>(rollingAccuracyAtomic.load() * 100.f);

            static int dataPointCounter = 0;
            ++dataPointCounter;

            constexpr int initialSaveNum = 500;
            constexpr int saveInterval = 10;

            if (dataPointCounter <= initialSaveNum || dataPointCounter % saveInterval == 0)
            {
                trainingHistory.push_back(dataPoint);
            }
        }

        // end of epoch
        double avgCost = epochCostSum / numBatches;
        LOG(LogLevel::INFO,
            "Epoch " + std::to_string(epoch + 1) +
            " of " + std::to_string(epochs) +
            " cost: " + std::to_string(avgCost));

        auto now = std::chrono::steady_clock::now();
        double thisEpochTime = std::chrono::duration<double>(now - trainingTimer.lastEpochTime).count();
        trainingTimer.lastEpochTime = now;

        // Accumulate for better average
        accumulatedEpochTime += thisEpochTime;
        epochCountForAverage++;

        double averageSoFar = accumulatedEpochTime / (double)epochCountForAverage;
        trainingTimer.epochDuration = averageSoFar; // store the rolling average

        if (stopRequestedAtomic.load())
        {
            break;
        }
    }

    stopRequestedAtomic.store(false); // reset for next time
}

void NeuralNetworkSubsystem::TrainOnMNISTFullProcess()
{
    PROFILE_LOG;

    if (!bIsMnistTrainingDataLoaded)
    {
        LOG(LogLevel::INFO, "No data loaded. Attempting auto-load from default paths...");

        bool loadedOk = LoadMNISTTrainingData(DEFAULT_TRAIN_IMAGES_PATH, DEFAULT_TRAIN_LABELS_PATH);
        if (!loadedOk)
        {
            LOG(LogLevel::ERROR, "Failed to auto-load MNIST training data from default path!");
            return;
        }
    }


    if (trainingInProgressAtomic.load())
    {
        LOG(LogLevel::WARNING, "Training is already in progress");
        return;
    }

    // start the training on async
    TrainOnMNISTAsync();
}

void NeuralNetworkSubsystem::TrainOnMNISTAsync()
{
    if (trainingInProgressAtomic.load())
    {
        LOG(LogLevel::WARNING, "Training in progress.");
        return;
    }

    // mark training as started
    trainingInProgressAtomic.store(true);
    stopRequestedAtomic.store(false);

    trainingTimer.startTime = std::chrono::steady_clock::now();
    trainingTimer.lastEpochTime = trainingTimer.startTime;
    trainingTimer.epochDuration = 0.0;
    trainingTimer.isInitialized = true;

    trainingThread = std::thread(&NeuralNetworkSubsystem::TrainOnMNISTThreadEntry, this);
    LOG(LogLevel::FLOW, "Launched async training thread.");
}

void NeuralNetworkSubsystem::StopTraining()
{
    if (trainingInProgressAtomic.load())
    {
        LOG(LogLevel::INFO, "Stop requested");
        stopRequestedAtomic.store(true);

        if (trainingThread.joinable())
        {
            trainingThread.join();
        }
        else
        {
            LOG(LogLevel::ERROR, "Failed to stop training thread.");
        }


        double finalElapsedTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - trainingTimer.startTime).count();
        trainingTimer.epochDuration = finalElapsedTime;
        trainingTimer.isInitialized = false; // stop the timer when we stop the training

        trainingInProgressAtomic.store(false);
        LOG(LogLevel::INFO, "Stop complete. Training thread joined.");
    }
    else
    {
        LOG(LogLevel::INFO, "No training in progress, nothing to stop idiot");
    }
}

bool NeuralNetworkSubsystem::SaveNetwork(const std::string& filePath)
{
    HyperParameters HyperParameters = HyperParameters::GetHyperParameters();

    bool bIsSuccessful = NeuralNetworkSerializer::SaveToJSON(
        filePath,
        CurrentNeuralNetwork,
        HyperParameters,
        trainingTimer,
        currentEpochAtomic.load(),
        totalEpochsAtomic.load(),
        trainingHistory
    );

    return bIsSuccessful;
}

bool NeuralNetworkSubsystem::LoadNetwork(const std::string& filePath)
{
    NeuralNetwork loadedNetwork;
    HyperParameters loadedHyperParameters;
    NeuralNetworkSubsystem::TrainingTimer loadedTimer;
    int loadedCurrentEpoch = 0;
    int loadedTotalEpochs = 0;
    std::vector<TrainingMetricPoint> loadedHistory;

    const bool bIsSuccessful = NeuralNetworkSerializer::LoadFromJSON(
        filePath,
        loadedNetwork,
        loadedHyperParameters,
        loadedTimer,
        loadedCurrentEpoch,
        loadedTotalEpochs,
        trainingHistory
    );

    if (!bIsSuccessful)
    {
        return false;
    }

    CurrentNeuralNetwork = std::move(loadedNetwork);

    if (CurrentNeuralNetwork.layers.empty())
    {
        LOG(LogLevel::WARNING, "Loaded checkpoint had 0 layers. Creating a default architecture...");
        InitNeuralNetwork(HyperParameters::activationType, HyperParameters::cost,
                          HyperParameters::defaultInputLayerSize,
                          HyperParameters::defaultNumHiddenLayers,
                          HyperParameters::defaultHiddenLayerSize,
                          HyperParameters::defaultOutputLayerSize);
    }

    bIsNeuralNetworkInitialized = true;

    HyperParameters::SetHyperParameters(loadedHyperParameters);

    trainingTimer = loadedTimer;

    currentEpochAtomic.store(loadedCurrentEpoch);
    totalEpochsAtomic.store(loadedTotalEpochs);

    {
        std::lock_guard<std::mutex> lock(metricMutex);
        trainingHistory = loadedHistory;
    }

    if (bIsMnistTrainingDataLoaded)
    {
        size_t datasetInputs = trainingDataSet.GetImages().empty() ? 0 : trainingDataSet.GetImages()[0].size();

        size_t networkInputs = CurrentNeuralNetwork.layers.empty() ? 0 : CurrentNeuralNetwork.layers[0].numNeuronsOutOfPreviousLayer;


        if (datasetInputs != networkInputs)
        {
            LOG(LogLevel::WARNING, "Potential mismatch: dataset input size = "
                + std::to_string(datasetInputs) + ", network input size = "
                + std::to_string(networkInputs));
        }
    }

    LOG(LogLevel::INFO, "Successfully loaded network from: " + filePath);

    return true;
}

std::pair<int, double> NeuralNetworkSubsystem::InferSingleImageFromPath(const std::string& path)
{
    std::vector<double> ProcessedImage = LoadAndProcessPNG(path);
    return InferSingleImage(ProcessedImage);
}

std::pair<int, double> NeuralNetworkSubsystem::InferSingleImage(const std::vector<double>& image)
{
    if (CurrentNeuralNetwork.layers.empty())
    {
        LOG(LogLevel::ERROR, "No layers in the network.. Inference impossible.");
        return {-1, 0};
    }

    std::vector<double> outputs = CurrentNeuralNetwork.ForwardPropagation(image, false);
    int bestIndex = -1;
    double bestValue = std::numeric_limits<double>::lowest();
    for (int i = 0; i < (int)outputs.size(); ++i)
    {
        LOG(LogLevel::DEBUG, "Output: " + std::to_string(i) + " > " + std::to_string(outputs[i]));
        if (outputs[i] > bestValue)
        {
            bestValue = outputs[i];
            bestIndex = i;
        }
    }
    LOG(LogLevel::INFO, "Inference Finished. Best Index (Predicted Digit): " + std::to_string(bestIndex) + " Best Value (Confidence in that digit): " + std::to_string(bestValue));
    return {bestIndex, bestValue};
}

std::vector<double> NeuralNetworkSubsystem::LoadAndProcessPNG(const std::string& path)
{
    int width, height, channels;

    // Load the image using stb_image
    unsigned char* imgData = stbi_load(path.c_str(), &width, &height, &channels, 1); // Force grayscale
    if (!imgData)
    {
        throw std::runtime_error("Failed to load image: " + path);
    }

    // Ensure the image is 28x28
    if (width != 28 || height != 28)
    {
        stbi_image_free(imgData);
        throw std::runtime_error("Image is not 28x28 pixels.");
    }

    std::vector<double> pixels;
    pixels.reserve(width * height);

    // Normalize pixel values to [0, 1]
    for (int i = 0; i < width * height; ++i)
    {
        pixels.push_back(static_cast<double>(imgData[i]) / 255.0);
    }

    // Free the image memory
    stbi_image_free(imgData);

    return pixels;
}

void NeuralNetworkSubsystem::SetVisualizationCallback(std::function<void(const NeuralNetwork&)> callback)
{
    visualizationCallback = std::move(callback);
}

void NeuralNetworkSubsystem::TrainOnMNISTThreadEntry()
{
    LOG(LogLevel::FLOW, "Started background training thread...");
    TrainOnMNIST();

    std::string FileName = "NeuralNetwork_" + NeuralNetworkUtility::GetInitTimestamp() + ".txt";
    std::string filePath = (std::filesystem::current_path() / "Saved" / FileName).string();
    // SaveNetwork(filePath);

    // StopTraining();
    trainingTimer.isInitialized = false; // stop the timer when we stop the training
    LOG(LogLevel::FLOW, "Finished background thread training. Network saved to: " + filePath);
}

// Multithreading stuffs
double NeuralNetworkSubsystem::SumDoubles(const std::vector<double>& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0);
}

void NeuralNetworkSubsystem::TestCustomSet()
{
    const std::filesystem::path testSetPath = std::filesystem::current_path() / "TrainingData" / "TestSet";
    if (!std::filesystem::exists(testSetPath) || !std::filesystem::is_directory(testSetPath))
    {
        LOG(LogLevel::ERROR, "Test set directory does not exist: " + testSetPath.string());
        return;
    }

    std::vector<std::string> imagePaths;
    for (const auto& entry : std::filesystem::directory_iterator(testSetPath))
    {
        if (entry.is_regular_file())
        {
            std::string ext = entry.path().extension().string();
            // Accept common image formats
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
            {
                imagePaths.push_back(entry.path().string());
            }
        }
    }

    if (imagePaths.empty())
    {
        LOG(LogLevel::ERROR, "No images found in test set directory: " + testSetPath.string());
        return;
    }

    // Prepare storage for images and their predictions
    testSetImages.clear();
    testSetPredictions.clear();

    int totalImages = static_cast<int>(imagePaths.size());
    LOG(LogLevel::INFO, "Found " + std::to_string(totalImages) + " images in the test set.");

    for (const auto& imgPath : imagePaths)
    {
        try
        {
            // Load and process image
            std::vector<double> processedImage = LoadAndProcessPNG(imgPath);

            // Perform inference
            std::pair<int, double> predictedDigit = InferSingleImage(processedImage);

            // Store image pixel data and prediction
            // Convert processedImage (std::vector<double>) to std::vector<float> for ImGui
            std::vector<float> imagePixels(processedImage.begin(), processedImage.end());

            testSetImages.push_back(imagePixels);
            testSetPredictions.push_back(predictedDigit.first);
            testSetConfidence.push_back(predictedDigit.second);
        }
        catch (const std::exception& e)
        {
            LOG(LogLevel::ERROR, "Failed to process image: " + imgPath + ". Error: " + e.what());
        }
    }

    LOG(LogLevel::INFO, "Completed inference on custom test set.");
}
