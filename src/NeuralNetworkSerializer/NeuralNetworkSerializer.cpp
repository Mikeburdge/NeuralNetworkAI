#include "NeuralNetworkSerializer.h"
#include <fstream>
#include "json.hpp"
#include "logging/Logger.h"

using json = nlohmann::json;

bool NeuralNetworkSerializer::SaveToJSON(const std::string& filePath,
                                         const NeuralNetwork& network,
                                         const HyperParameters& hyperParams,
                                         const NeuralNetworkSubsystem::TrainingTimer& trainingTimer,
                                         int currentEpoch,
                                         int totalEpochs,
                                         const std::vector<NeuralNetworkSubsystem::TrainingMetricPoint>& trainingHistory)
{
    json root;

    // 1) Store all HyperParameters
    root["HyperParameters"] = {
        {"learningRate", hyperParams.learningRate},
        {"batchSize", hyperParams.batchSize},
        {"epochs", hyperParams.epochs},
        {"momentum", hyperParams.momentum},
        {"weightDecay", hyperParams.weightDecay},
        {"useDropoutRate", hyperParams.useDropoutRate},
        {"dropoutRate", hyperParams.dropoutRate},
        {"costType", hyperParams.cost},
        {"activationType", hyperParams.activationType}
    };

    // 2) Store the training
    root["TrainingState"] = {
        {"currentEpoch", currentEpoch},
        {"totalEpochs", totalEpochs},
        {"isTimerInit", trainingTimer.isInitialized},
        {"elapsedTime", std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - trainingTimer.startTime).count()},
        {"timerEpochDuration", trainingTimer.epochDuration}
    };

    // 3) store trainingHistory to recreate the graph
    {
        json historyArray = json::array();
        for (auto& point : trainingHistory)
        {
            json pt;
            pt["timeSeconds"] = point.timeSeconds;
            pt["loss"] = point.loss;
            pt["accuracy"] = point.accuracy;
            pt["rollingAcc"] = point.rollingAcc;
            historyArray.push_back(pt);
        }
        root["trainingHistory"] = historyArray;
    }

    // 4) Layers
    json layersJson = json::array();
    for (auto& layer : network.layers)
    {
        json layerObj;
        layerObj["activation"] = layer.activation;
        layerObj["cost"] = layer.cost;
        layerObj["numNeurons"] = layer.numNeurons;
        layerObj["numNeuronsOutOfPreviousLayer"] = layer.numNeuronsOutOfPreviousLayer;

        // biases
        layerObj["biases"] = layer.biases;

        // velocity
        // velocity is a 2D array similar to weights
        {
            bool bAreAllZero = true;
            json velMatrix = json::array();
            for (auto& velRow : layer.velocity)
            {
                for (auto& velocity : velRow)
                {
                    if (fabs(velocity) > 1e-12)
                    {
                        bAreAllZero = false;
                        break;
                    }
                }
                if (!bAreAllZero)
                {
                    break;
                }
            }
            if (!bAreAllZero)
            {
                // Write the full velocity
                json velMatrix = json::array();
                for (auto& velRow : layer.velocity)
                    velMatrix.push_back(velRow);

                layerObj["velocity"] = velMatrix;
                layerObj["velocitySkipped"] = false;
            }
            else
            {
                // Indicate we intentionally skip zero velocity
                layerObj["velocitySkipped"] = true;
            }
        }

        // weights
        json weightMatrix = json::array();
        for (auto& row : layer.weights)
        {
            weightMatrix.push_back(row);
        }
        layerObj["weights"] = weightMatrix;

        layersJson.push_back(layerObj);
    }
    root["Layers"] = layersJson;

    // 5) Write to file
    try
    {
        std::ofstream ofs(filePath);
        if (!ofs.is_open())
        {
            LOG(LogLevel::ERROR, "Cannot open file for saving: " + filePath);
            return false;
        }
        ofs << root.dump(4); // pretty-print with 4-space indentation
        ofs.close();
        LOG(LogLevel::INFO, "Network saved to JSON: " + filePath);
        return true;
    }
    catch (const std::exception& e)
    {
        LOG(LogLevel::ERROR, e.what());
        return false;
    }
}

bool NeuralNetworkSerializer::LoadFromJSON(const std::string& filePath,
                                           NeuralNetwork& outNetwork,
                                           HyperParameters& outHyperParams,
                                           NeuralNetworkSubsystem::TrainingTimer& outTimer,
                                           int& outCurrentEpoch,
                                           int& outTotalEpochs,
                                           std::vector<NeuralNetworkSubsystem::TrainingMetricPoint>& outTrainingHistory)
{
    try
    {
        std::ifstream ifs(filePath);

        if (!ifs.is_open())
        {
            LOG(LogLevel::ERROR, "Cannot open file for loading: " + filePath);
            return false;
        }

        json root;
        ifs >> root;
        ifs.close();

        // Hyper parameters
        auto hyperParams = root["HyperParameters"];
        outHyperParams.learningRate = hyperParams["learningRate"];
        outHyperParams.batchSize = hyperParams["batchSize"];
        outHyperParams.epochs = hyperParams["epochs"];
        outHyperParams.momentum = hyperParams["momentum"];
        outHyperParams.weightDecay = hyperParams["weightDecay"];
        outHyperParams.useDropoutRate = hyperParams["useDropoutRate"];
        outHyperParams.dropoutRate = hyperParams["dropoutRate"];
        outHyperParams.cost = (CostType)hyperParams["costType"];
        outHyperParams.activationType = (ActivationType)hyperParams["activationType"];

        // training 
        auto trainingState = root["TrainingState"];
        outCurrentEpoch = trainingState["currentEpoch"];
        outTotalEpochs = trainingState["totalEpochs"];
        bool isTimerInit = trainingState["isTimerInit"];
        outTimer.isInitialized = isTimerInit;
        outTimer.epochDuration = trainingState["timerEpochDuration"];
        float elapsedTime = trainingState["elapsedTime"];

        std::chrono::duration<float> elapsedDuration(elapsedTime);
        std::chrono::steady_clock::duration steadyElapsed = std::chrono::duration_cast<std::chrono::steady_clock::duration>(elapsedDuration);
        outTimer.startTime = std::chrono::steady_clock::now() - steadyElapsed;

        // todo: Training history,
        // probably gonna store periodically, maybe proportionally to the number of epochs trained. Otherwise it will be far too chonky for higher epoch network.
        // This will modify the location/ frequency of where its being stored for these training data packs as opposed to only saving a proportionally smaller number.
        // I will make it start at a decently high number of epochs to maintain the visual aspect of the training when the learning should be at its highest.
        outTrainingHistory.clear();
        auto historyArray = root["TrainingHistory"];
        for (nlohmann::basic_json<> item : historyArray)
        {
            NeuralNetworkSubsystem::TrainingMetricPoint point;
            point.timeSeconds = item["timeSeconds"];
            point.loss = item["loss"];
            point.accuracy = item["accuracy"];
            point.rollingAcc = item["rollingAcc"];
            outTrainingHistory.push_back(point);
        }

        // Layers
        outNetwork.layers.clear();
        auto layersJson = root["Layers"];
        for (auto& layer : layersJson)
        {
            ActivationType activationType = layer["activation"];
            CostType costType = layer["cost"];
            int numNeurons = layer["numNeurons"];
            int numNeuronsOutOfPreviousLayer = layer["numNeuronsOutOfPreviousLayer"];

            Layer newLayer(activationType, costType, numNeurons, numNeuronsOutOfPreviousLayer);

            newLayer.biases = layer["biases"].get<std::vector<double>>();

            auto weightMatrix = layer["weights"];
            newLayer.weights.resize(weightMatrix.size());
            for (size_t t = 0; t < weightMatrix; t++)
            {
                newLayer.weights.push_back(weightMatrix[t]);
            }

            if (layer.contains("velocitySkipped") && layer["velocitySkipped"].get<bool>() == true)
            {
                newLayer.velocity.resize(numNeurons);
                for (size_t i = 0; i < numNeurons; ++i)
                {
                    newLayer.velocity[i].resize(numNeuronsOutOfPreviousLayer, 0.0);
                }
            }
            else
            {
                auto velocityMatrix = layer["velocity"];
                newLayer.velocity.resize(velocityMatrix.size());

                for (size_t t = 0; t < velocityMatrix; t++)
                {
                    newLayer.velocity[t] = velocityMatrix[t].get<std::vector<double>>();
                }
            }

            newLayer.neurons.resize(numNeurons);

            outNetwork.layers.push_back(newLayer);
        }
        LOG(LogLevel::INFO, "Loaded network from JSON: " + filePath);
        return true;
    }
    catch (std::exception e)
    {
        LOG(LogLevel::ERROR, e.what());
        return false;
    }
}
