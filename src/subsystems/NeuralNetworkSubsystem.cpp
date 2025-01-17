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


    // Main Training Loop
    totalEpochsAtomic.store(epochs);

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        currentEpochAtomic.store(epoch);

        double epochAverageCost = 0.0;
        double epochAccuracy = 0.0;

        currentLossAtomic.store(float(epochAverageCost));
        currentAccuracyAtomic.store(float(epochAccuracy));

        // Shuffle the dataset indeces. its not necessary but its normal for Stochastic Gradient Descent
        {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        if (stopRequested.load())
        {
            LOG(LogLevel::INFO, "Early Stop: user requested stop at epoch " + std::to_string(epoch));
            break;
        }

        double epochCostSum = 0.0; // accumulate cost accross mini-batches
        int numBatches = 0;
        // mini batches
        for (size_t startIndex = 0; startIndex < datasetSize; startIndex += batchsize)
        {
            if (stopRequested.load())
            {
                LOG(LogLevel::INFO, "Stop mid-epoch...");
                goto doneTraining; // I dont like doing goto's and I havent done one in over a decade but fck it
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

            ++numBatches;
            epochCostSum += batchCost; // from your original code
        }

        // end of epoch
        double avgCost = epochCostSum / numBatches;
        LOG(LogLevel::INFO, "Epoch " + std::to_string(epoch) + " cost= " + std::to_string(avgCost));
    }

doneTraining:
    stopRequested.store(false); // reset for next time
    LOG(LogLevel::INFO, "TrainOnMNIST done or stopped early.");
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
    if (CurrentNeuralNetwork.layers.empty())
    {
        LOG(LogLevel::INFO, "No existing network. Auto-creating 784->128->10 with Sigmoid/CrossEntropy.");

        // todo: After some testing I want to see if adding a second hidden layer produces better results
        InitNeuralNetwork(sigmoid, crossEntropy, /*input*/ 784, /*hiddenLayers*/ 1, /*HiddenLayerSize*/ 128, /*output*/ 10);
    }

    if (trainingInProgress.load())
    {
        LOG(LogLevel::WARNING, "Training is already in progress");
        return;
    }

    // start the training
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

    trainingThread = std::thread(&NeuralNetworkSubsystem::TrainOnMNISTThreadEntry, this);
    LOG(LogLevel::FLOW, "Launched async training thread.");
}

void NeuralNetworkSubsystem::StopTraining()
{
    if (trainingInProgress.load())
    {
        LOG(LogLevel::INFO, "Stop requested, NOT IMPLEMENTED");
        // ADDING THERE HERE BUT ITS NOT FINISHED YET
    }
}

bool NeuralNetworkSubsystem::SaveNetwork(const std::string& filePath)
{
    // todo: could and probably should make a new subsystem/ file for this but i need this quickly so fuck it
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
    NeuralNetwork newNeuralNetwork;
    int layerCount;
    ifs >> layerCount;

    for (int i = 0; i < layerCount; ++i)
    {
        int numNeurons, outOfPreviousLayer;
        ifs >> numNeurons >> outOfPreviousLayer;
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
                ifs >> layer.weights[weightsMatrixIndex][weightsIndex];
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
