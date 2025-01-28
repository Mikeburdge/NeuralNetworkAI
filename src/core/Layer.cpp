#include "Layer.h"

#include <random>

#include "Activation.h"
#include "HyperParameters.h"
#include "vector"
#include "../Logging/Logger.h"

#pragma optimize("", off)

using namespace std;

vector<double> Layer::computeOutput(const vector<double>& input)
{
    PROFILE_LOG;
    vector<double> output(numNeurons, 0.0);

    for (int neuronsIn = 0; neuronsIn < numNeurons; ++neuronsIn)
    {
        double neuronOutput = biases[neuronsIn]; // Start with the bias

        for (int neuronsOut = 0; neuronsOut < numNeuronsOutOfPreviousLayer; ++neuronsOut)
        {
            neuronOutput += input[neuronsOut] * weights[neuronsIn][neuronsOut]; // Sum the weighted inputs
        }
        output[neuronsIn] = neuronOutput; // Make sure we store the output
    }

    if (activation == ActivationType::softmax)
    {
        // Exponentiate the raw values then normalize to get probabilities
        double maxRaw = *std::max_element(output.begin(), output.end()); // numerical stability
        double sumExp = 0.0;
        for (int i = 0; i < numNeurons; ++i)
        {
            double expVal = std::exp(output[i] - maxRaw);
            output[i] = expVal;
            sumExp += expVal;
        }

        for (int i = 0; i < numNeurons; ++i)
        {
            double softmaxVal = output[i] / sumExp;
            neurons[i].ActivationValue = output[i] = softmaxVal;
        }
    }
    else
    {
        for (int neuronIn = 0; neuronIn < numNeurons; ++neuronIn)
        {
            double activatedVal = Activation::CalculateActivation(activation, output[neuronIn]);
            neurons[neuronIn].ActivationValue = output[neuronIn] = activatedVal;
        }
    }

    return output;
}

void Layer::adjustWeights(const vector<double>& errorGradient, const std::vector<double>& prevLayerActivations)
{
    PROFILE_LOG;

    const double learningRate = HyperParameters::learningRate;

    // Update weights based on the error gradient
    for (int i = 0; i < numNeurons; ++i)
    {
        for (int j = 0; j < weights[i].size(); ++j)
        {
            // weights[i][j] -= learningRate * errorGradient[i] * prevLayerActivations[j];

            const double gradientTerm = errorGradient[i] * prevLayerActivations[j];
            velocity[i][j] = (HyperParameters::momentum * velocity[i][j]) + (HyperParameters::learningRate * gradientTerm);

            const double decay = HyperParameters::weightDecay * weights[i][j];

            weights[i][j] -= (velocity[i][j] + decay);
        }

        // string logMessage = "Iteration " + to_string(i) + " of " + to_string(numNeurons) + " through neurons";
        // LOG(LogLevel::DEBUG, logMessage);

        biases[i] -= learningRate * errorGradient[i]; // Update biases
    }
}

void Layer::InitializeRandomBiases(mt19937& rng)
{
    PROFILE_LOG;

    normal_distribution<double> distribution(0.0, 1.0);

    for (double& bias : biases)
    {
        const double randValue = distribution(rng) / sqrt(biases.size());
        bias = randValue;
    }
}

void Layer::InitializeRandomWeights(mt19937& rng)
{
    PROFILE_LOG;
    normal_distribution<double> distribution(0.0, 1.0);

    // He Initialization: stddev = sqrt(2 / fan_in)
    double stddev = sqrt(2.0 / static_cast<double>(numNeuronsOutOfPreviousLayer));

    for (int i = 0; i < numNeurons; i++)
    {
        for (int j = 0; j < numNeuronsOutOfPreviousLayer; j++)
        {
            const double randValue = distribution(rng) * stddev;
            weights[i][j] = randValue;
        }
    }
}

vector<double> Layer::CalculatePreviousLayerError(const vector<double>& currentLayersErrorGradient, const vector<Neuron>& previousLayerNeurons) const
{
    PROFILE_LOG;
    vector<double> previousLayerErrorGradient(numNeuronsOutOfPreviousLayer, 0.0);

    for (int prevNeuronIdx = 0; prevNeuronIdx < numNeuronsOutOfPreviousLayer; ++prevNeuronIdx)
    {
        double error = 0.0f;

        for (int neuronIdx = 0; neuronIdx < numNeurons; ++neuronIdx)
        {
            error += weights[neuronIdx][prevNeuronIdx] * currentLayersErrorGradient[neuronIdx];
        }

        double activatedValue = previousLayerNeurons[prevNeuronIdx].ActivationValue;

        // calculate if we're in the final layer, would be nice to pass this in for a more guaranteed
        // final layer but it should be fine as we only ever choose sfoftmax for final layer. 
        bool bIsFinalLayer = (cost == crossEntropy && activation == softmax);

        double derivative = bIsFinalLayer ? 1.0 : Activation::CalculateActivationDerivative(activation, activatedValue);

        previousLayerErrorGradient[prevNeuronIdx] = error * derivative;
    }

    return previousLayerErrorGradient;
}

void Layer::SetDropout(bool useDropoutRate, float dropoutRate)
{
    // todo: To Be Implemented
}
