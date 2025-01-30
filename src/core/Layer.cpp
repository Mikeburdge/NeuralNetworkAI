#include "Layer.h"

#include <random>

#include "Activation.h"
#include "HyperParameters.h"
#include "vector"
#include "../Logging/Logger.h"

#pragma optimize("", off)

using namespace std;

void Layer::InitAdam()
{
    m.resize(numNeurons, std::vector<double>(numNeuronsOutOfPreviousLayer, 0.0));
    v.resize(numNeurons, std::vector<double>(numNeuronsOutOfPreviousLayer, 0.0));

    mBias.resize(numNeurons, 0.0);
    vBias.resize(numNeurons, 0.0);
}

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

    if (useDropout)
    {
        random_device rd;
        mt19937 rng(rd());
        ApplyDropout(rng);
    }

    return output;
}

void Layer::adjustWeights(const std::vector<double>& errorGradient, const std::vector<double>& prevLayerActivations)
{
    PROFILE_LOG;

    t++; // increment time step.

    // set the adam hyperparametyers.
    // for the sake of copying from an equation I'm using their notation terms
    const double alpha = HyperParameters::learningRate;
    const double beta1 = this->beta1;
    const double beta2 = this->beta2;
    const double epsilon = this->epsilon;
    const double weightDecay = HyperParameters::weightDecay;

    // Update weights based on the error gradient
    for (int i = 0; i < numNeurons; ++i)
    {
        for (int j = 0; j < weights[i].size(); ++j)
        {
            const double gradient = errorGradient[i] * prevLayerActivations[j];

            m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradient;
            v[i][j] = beta2 * v[i][j] + (1 - beta2) * (gradient * gradient);

            const double mHat = m[i][j] / (1 - pow(beta1, t));
            const double vHat = v[i][j] / (1 - pow(beta2, t));

            weights[i][j] -= alpha * (mHat / (sqrt(vHat) + epsilon) + weightDecay * weights[i][j]);

        }

        biases[i] -= alpha * errorGradient[i]; // Update biases

        const double biasGradient = errorGradient[i];
        mBias[i] = beta1 * mBias[i] + (1 - beta1) * biasGradient;
        vBias[i] = beta2 * vBias[i] + (1 - beta2) * (biasGradient * biasGradient);

        const double mBiasHat = mBias[i] / (1 - pow(beta1, t));
        const double vBiasHat = vBias[i] / (1 - pow(beta2, t));

        biases[i] -= alpha * (mBiasHat / (sqrt(vBiasHat) + epsilon) + weightDecay * biases[i]);
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
    this->useDropout = useDropoutRate;
    this->dropoutRate = dropoutRate;
    if (useDropoutRate)
    {
        InitializeDropoutMask();
    }
}

void Layer::InitializeDropoutMask()
{
    dropoutMask.resize(numNeurons, false);
}

void Layer::ApplyDropout(mt19937& rng)
{
    if (!useDropout)
        return;

    // Create a Bernoulli distribution for dropout
    bernoulli_distribution dropoutDist(dropoutRate);

    for (int i = 0; i < numNeurons; ++i)
    {
        dropoutMask[i] = dropoutDist(rng);
        if (dropoutMask[i])
        {
            neurons[i].ActivationValue = 0.0; // Drop the neuron by setting its activation to zero
        }
    }
}
