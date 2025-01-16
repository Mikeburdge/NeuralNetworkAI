#include "NeuralNetworkSubsystem.h"

#include <iostream>
#include <utility>
#include <vulkan/vulkan_core.h>
#include <core/HyperParameters.h>
#include <logging/Logger.h>


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
    mnistTrainingDataLoaded = loaded;

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
    if (!mnistTrainingDataLoaded)
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

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        // Shuffle the dataset indeces. its not necessary but its normal for Stochastic Gradient Descent
        {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        double epochCostSum = 0.0; // accumulate cost accross mini-batches
        int numBatches = 0;
        // mini batches
        for (size_t startIndex = 0; startIndex < datasetSize; startIndex += batchsize)
        {
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

            if (visualizationCallback)
            {
                visualizationCallback(CurrentNeuralNetwork);
            }
        } // End of mini batch loop
        double epochCostAvg = epochCostSum / (double)numBatches;

        LOG(LogLevel::INFO, "Epoch %d / %d batches - AverageCost: %", epoch, (double)numBatches, epochCostAvg);
    }
    LOG(LogLevel::INFO, "Completed mini-batch training on MNIST!");
}

void NeuralNetworkSubsystem::TrainOnMNISTFullProcess()
{
    PROFILE_LOG;

    if (!mnistTrainingDataLoaded)
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

    // start the training
    TrainOnMNIST();
}

void NeuralNetworkSubsystem::SetVisualizationCallback(std::function<void(const NeuralNetwork&)> callback)
{
    visualizationCallback = std::move(callback);
}
