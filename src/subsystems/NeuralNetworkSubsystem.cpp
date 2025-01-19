#include "NeuralNetworkSubsystem.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <core/HyperParameters.h>
#include <logging/Logger.h>
#include <vulkan/vulkan_core.h>

// Include stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "utility/NeuralNetworkUtility.h"

NeuralNetworkSubsystem::TrainingTimer NeuralNetworkSubsystem::trainingTimer;

void NeuralNetworkSubsystem::InitNeuralNetwork(const ActivationType& inActivation, const CostType& inCost,
                                               const int inputLayerSize,
                                               const int hiddenLayers, const int hiddenLayersSizes,
                                               const int outputLayerSize)
{
    CurrentNeuralNetwork = NeuralNetwork();

    HyperParameters::cost = inCost;
    HyperParameters::activationType = inActivation;

    // Reserve the input, output and hidden layers
    CurrentNeuralNetwork.layers.reserve(hiddenLayers + 2);


    // For now as every hidden layer has the same number of neurons we can just pass the hiddenLayersSizes in as the out neurons
    const Layer inputLayer(inActivation, inCost, inputLayerSize, 0);

    CurrentNeuralNetwork.AddLayer(inputLayer);


    for (int i = 0; i < hiddenLayers; i++)
    {
        int neuronsOut;
        if (i == 0)
        {
            neuronsOut = inputLayerSize; // Nodes coming out of the last layer
        }
        else
        {
            neuronsOut = hiddenLayersSizes;
        }
        const Layer hiddenLayer(inActivation, inCost, hiddenLayersSizes, neuronsOut);

        CurrentNeuralNetwork.AddLayer(hiddenLayer);
    }

    const Layer outputLayer(inActivation, inCost, outputLayerSize, hiddenLayersSizes);

    CurrentNeuralNetwork.AddLayer(outputLayer);
}

NeuralNetwork& NeuralNetworkSubsystem::GetNeuralNetwork()
{
    return CurrentNeuralNetwork;
}

void NeuralNetworkSubsystem::StartNeuralNetwork(const std::vector<double>& inputData,
                                                const std::vector<double>& targetOutput)
{
    PROFILE_LOG;
    NeuralNetwork& network = GetNeuralNetwork();

    const CostType currentCost = HyperParameters::cost;

    for (int epoch = 0; epoch < HyperParameters::epochs; ++epoch)
    {
        // Perform forward propagation
        std::vector<double> predictions = network.ForwardPropagation(inputData);

        // Calculate Cost and Log Progress
        double cost = Cost::CalculateCost(currentCost, predictions, targetOutput);

        std::cout << "Epoch " << epoch << " - Cost: " << cost << '\n';

        // Calculate Error Gradient for Backpropogation
        std::vector<double> costGradient = Cost::CalculateCostDerivative(currentCost, predictions, targetOutput);

        // Perform Backwards Propagation
        network.BackwardPropagation(costGradient);

        // if (epoch % ::HyperParameters::visualizationUpdateInterval == 0 &&  visualizationCallback)
        {
            visualizationCallback(CurrentNeuralNetwork);
        }
    }

    // const int outputLayerIndex = network.layers.size() - 1;
    //
    // Layer outputLayer = CurrentNeuralNetwork.layers[outputLayerIndex];
    //Cost::CalculateCost HERE LAST

    // Example: Print or process the output

    // Further actions after the propagation can be added here
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

void NeuralNetworkSubsystem::TrainOnMNIST()
{
    PROFILE_LOG;

    // Neeeds to have data
    if (!bIsMnistTrainingDataLoaded)
    {
        LOG(LogLevel::ERROR, "Failed to load MNIST training data!");
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


    int epochs = HyperParameters::epochs;
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
        // todo: Need to re-add and get this function working properly 
        // layer.SetDropout(HyperParameters::useDropoutRate, HyperParameters::dropoutRate);
    }

    // Main Training Loop
    totalEpochsAtomic.store(epochs);

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        currentEpochAtomic.store(epoch);


        // Shuffle dataset
        std::shuffle(indices.begin(), indices.end(), rng);

        // Calculate total batches for this epoch
        size_t totalBatches = (datasetSize + batchsize - 1) / batchsize; // Ceiling division

        double epochAverageCost = 0.0;
        double epochAccuracy = 0.0;

        currentLossAtomic.store(float(epochAverageCost));
        currentAccuracyAtomic.store(float(epochAccuracy));
        totalBatchesInEpoch.store(static_cast<int>(totalBatches));

        // Reset current batch index
        currentBatchIndex.store(0);

        // Initialize epoch variables
        double epochCostSum = 0.0;
        int numBatches = 0;

        for (size_t startIndex = 0; startIndex < datasetSize; startIndex += batchsize)
        {
            if (stopRequested.load())
            {
                LOG(LogLevel::INFO, "Stop mid-epoch...");
                break;
            }

            size_t endIndex = std::min(startIndex + batchsize, datasetSize); // Again possibly datasetSize - 1 if we hit out of index stuff here
            size_t realBatch = endIndex - startIndex;
            // make the batch
            std::vector<std::vector<double>> batchInputs(realBatch);
            std::vector<std::vector<double>> batchTargets(realBatch);

            for (size_t i = 0; i < realBatch; ++i)
            {
                size_t idx = indices[startIndex + i];
                batchInputs[i] = dataset.GetImages()[idx];
                batchTargets[i] = dataset.GetLabelsOneHot()[idx];
            }

            // Forward propogation, accumulating cost
            double batchCost = 0.0;
            for (size_t i = 0; i < realBatch; ++i)
            {
                std::vector<double> predictions = network.ForwardPropagation(batchInputs[i]);
                batchCost += Cost::CalculateCost(currentCost, predictions, batchTargets[i]);
            }
            batchCost /= (double)realBatch;
            epochCostSum += batchCost;
            ++numBatches;


            // Cost Geadient stuff for each sample
            // Theres a chance to do a better, more sophisticated 
            std::vector<double> totalGradient(network.layers.back().numNeurons, 0.0);

            for (size_t i = 0; i < realBatch; ++i)
            {
                // Could do it differently here too but this is simpler for now.
                std::vector<double> predictions = network.ForwardPropagation(batchInputs[i]);
                std::vector<double> costGradient = Cost::CalculateCostDerivative(currentCost, predictions, batchTargets[i]);

                // Add them all up
                for (size_t j = 0; j < totalGradient.size(); ++j)
                {
                    totalGradient[j] += costGradient[j];
                }
            }

            // Average the gradient
            for (size_t j = 0; j < totalGradient.size(); ++j)
            {
                totalGradient[j] /= (double)realBatch;
            }

            for (size_t j = 0; j < totalGradient.size(); ++j)
            {
                // Use backpropogation on the averaghe gradient
                network.BackwardPropagation(totalGradient);
            }

            // update the visuals

            vizBatchCounter++;
            if (vizBatchCounter % vizUpdateInterval == 0)
            {
                if (visualizationCallback)
                    visualizationCallback(CurrentNeuralNetwork);
            }

            // Increment current batch index
            currentBatchIndex.fetch_add(1);
        }
        // Calculate epoch duration
        std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
        trainingTimer.epochDuration = std::chrono::duration<double>(now - trainingTimer.lastEpochTime).count();
        trainingTimer.lastEpochTime = now;

        // end of epoch
        double avgCost = epochCostSum / numBatches;
        LOG(LogLevel::INFO,
            "Epoch " + std::to_string(epoch + 1) +
            " of " + std::to_string(epochs) +
            " cost: " + std::to_string(avgCost));

        // update the UI
        currentLossAtomic.store(static_cast<float>(avgCost));
        currentAccuracyAtomic.store(static_cast<float>(epochAccuracy));

        if (stopRequested.load())
        {
            break;
        }
    }

    stopRequested.store(false); // reset for next time
}


void NeuralNetworkSubsystem::TrainOnMNISTFullProcess()
{
    PROFILE_LOG;

    if (!bIsMnistTrainingDataLoaded)
    {
        LOG(LogLevel::INFO, "No data loaded. Attempting auto-load from default paths...");

        bool loadedOk = LoadMNISTTrainingData(DEFAULT_IMAGES_PATH, DEFAULT_LABELS_PATH);
        if (!loadedOk)
        {
            LOG(LogLevel::ERROR, "Failed to auto-load MNIST training data from default path!");
            return;
        }
    }


    if (trainingInProgress.load())
    {
        LOG(LogLevel::WARNING, "Training is already in progress");
        return;
    }

    // start the training on async
    TrainOnMNISTAsync();
}

void NeuralNetworkSubsystem::TrainOnMNISTAsync()
{
    if (trainingInProgress.load())
    {
        LOG(LogLevel::WARNING, "Training in progress.");
        return;
    }

    // mark training as started
    trainingInProgress.store(true);
    stopRequested.store(false);

    trainingTimer.startTime = std::chrono::steady_clock::now();
    trainingTimer.lastEpochTime = trainingTimer.startTime;
    trainingTimer.epochDuration = 0.0;
    trainingTimer.isInitialized = true;

    trainingThread = std::thread(&NeuralNetworkSubsystem::TrainOnMNISTThreadEntry, this);
    LOG(LogLevel::FLOW, "Launched async training thread.");
}

void NeuralNetworkSubsystem::StopTraining()
{
    if (trainingInProgress.load())
    {
        LOG(LogLevel::INFO, "Stop requested");
        stopRequested.store(true);

        if (trainingThread.joinable())
        {
            trainingThread.join();
        }
        else
        {
            LOG(LogLevel::ERROR, "Failed to stop training thread.");
        }

        LOG(LogLevel::INFO, "Stop complete. Training thread joined.");
    }
    else
    {
        LOG(LogLevel::INFO, "No training in progress, nothing to stop idiot");
    }
}

bool NeuralNetworkSubsystem::SaveNetwork(const std::string& filePath)
{
    // todo: could and probably should make a new subsystem/ file for this but i need this quickly so fuck it
    try
    {
        std::filesystem::path path = filePath;
        std::filesystem::path parentDir = path.parent_path();
        if (parentDir.empty() && !std::filesystem::exists(parentDir))
        {
            std::filesystem::create_directories(parentDir);
        }
    }
    catch (std::exception& e)
    {
        LOG(LogLevel::ERROR, "Failed to create directories for saving the network: " + std::string(e.what()));
        return false;
    }

    std::ofstream ofs(filePath);
    if (!ofs.is_open())
    {
        LOG(LogLevel::ERROR, "Could not open file to save: " + filePath);
        // LOG(LogLevel::ERROR, "Could not open file to save: " + filePath);
        return false;
    }

    int layerCount = (int)CurrentNeuralNetwork.layers.size();
    ofs << layerCount << "\n";
    for (Layer layer : CurrentNeuralNetwork.layers)
    {
        // neurons
        ofs << layer.numNeurons << " " << layer.numNeuronsOutOfPreviousLayer << "\n";

        // biases
        for (double bias : layer.biases)
        {
            ofs << bias << " ";
        }
        // next line after biases
        ofs << "\n";

        // weights

        for (std::vector<double> weightVector : layer.weights)
        {
            for (double weight : weightVector)
            {
                ofs << weight << " ";
            }
            ofs << "\n";
        }
    }
    LOG(LogLevel::INFO, "Saved network to: " + filePath);
    return true;
}

bool NeuralNetworkSubsystem::LoadNetwork(const std::string& filePath)
{
    std::ifstream ifs(filePath);
    if (!ifs.is_open())
    {
        LOG(LogLevel::ERROR, "Could not open file to load: " + filePath);
        return false;
    }

    int layerCount = 0;
    if (!(ifs >> layerCount))
    {
        LOG(LogLevel::ERROR, "Invalid network file format (layerCount).");
        return false;
    }

    NeuralNetwork newNeuralNetwork;

    for (int i = 0; i < layerCount; ++i)
    {
        int numNeurons, outOfPreviousLayer;

        if (!(ifs >> numNeurons >> outOfPreviousLayer))
        {
            LOG(LogLevel::ERROR, "Malformed network file (layer header).");
            return false;
        }

        Layer layer(HyperParameters::activationType, HyperParameters::cost, numNeurons, outOfPreviousLayer);

        // biases
        for (int biasesIndex = 0; biasesIndex < numNeurons; ++biasesIndex)
        {
            ifs >> layer.biases[biasesIndex];
        }

        // weights
        for (int weightsMatrixIndex = 0; weightsMatrixIndex < numNeurons; ++weightsMatrixIndex)
        {
            for (int weightsIndex = 0; weightsIndex < layer.numNeuronsOutOfPreviousLayer; ++weightsIndex)
            {
                if (!(ifs >> layer.weights[weightsMatrixIndex][weightsIndex]))
                {
                    LOG(LogLevel::ERROR, "Malformed network file (weight read).");
                    return false;
                }
            }
        }
        newNeuralNetwork.layers.push_back(layer);
    }
    CurrentNeuralNetwork = newNeuralNetwork;
    LOG(LogLevel::INFO, "Loaded network from: " + filePath);
    return true;
}

int NeuralNetworkSubsystem::InferSingleImageFromPath(const std::string& path)
{
    std::vector<double> ProcessedImage = LoadAndProcessPNG(path);
    return InferSingleImage(ProcessedImage);
}

int NeuralNetworkSubsystem::InferSingleImage(const std::vector<double>& image)
{
    if (CurrentNeuralNetwork.layers.empty())
    {
        LOG(LogLevel::ERROR, "No layers in the network.. Inference impossible.");
        return false;
    }

    std::vector<double> outputs = CurrentNeuralNetwork.ForwardPropagation(image);
    int bestIndex = -1;
    double bestValue = std::numeric_limits<double>::lowest();
    for (int i = 0; i < (int)outputs.size(); ++i)
    {
        if (outputs[i] > bestValue)
        {
            bestValue = outputs[i];
            bestIndex = i;
        }
    }
    LOG(LogLevel::INFO, "Inference Finished. Best Index (Predicted Digit): " + std::to_string(bestIndex) + " Best Value (Confidence in that digit): " + std::to_string(bestValue));
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
    SaveNetwork(filePath);

    trainingInProgress.store(false);
    LOG(LogLevel::FLOW, "Finished background thread training. Network saved to: " + filePath);
}
