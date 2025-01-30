#pragma once

#include "HyperParameters.h"

#include "Activation.h"

int HyperParameters::defaultInputLayerSize = 784;
int HyperParameters::defaultNumHiddenLayers = 2;
int HyperParameters::defaultHiddenLayerSize = 256; // 128 / 256
int HyperParameters::defaultOutputLayerSize = 10;

float HyperParameters::defaultLearningRate = HyperParameters::learningRate = 0.001f;
int HyperParameters::defaultBatchSize = HyperParameters::batchSize = 64;
int HyperParameters::defaultEpochs = HyperParameters::epochs = 10;

double HyperParameters::defaultWeightDecay = HyperParameters::weightDecay = 0.0005;
bool HyperParameters::defaultUseDropoutRate = HyperParameters::useDropoutRate = true;
float HyperParameters::defaultDropoutRate = HyperParameters::dropoutRate = 0.2f;
bool HyperParameters::defaultUseGradientClipping = HyperParameters::useGradientClipping = true;
double HyperParameters::defaultGradientClipThreshold = HyperParameters::gradientClipThreshold = 1.0;

CostType HyperParameters::cost = crossEntropy;
ActivationType HyperParameters::activationType = ReLU;

int HyperParameters::visualizationUpdateInterval = 5;

void HyperParameters::ResetHyperParameters()
{
    learningRate = defaultLearningRate;
    batchSize = defaultBatchSize;
    epochs = defaultEpochs;

    weightDecay = defaultWeightDecay;
    useDropoutRate = defaultUseDropoutRate;
    dropoutRate = defaultDropoutRate;
}

void HyperParameters::SetHyperParameters(HyperParameters hyperParameters)
{
    learningRate = hyperParameters.learningRate;
    batchSize = hyperParameters.batchSize;
    epochs = hyperParameters.epochs;
    weightDecay = hyperParameters.weightDecay;
    useDropoutRate = hyperParameters.useDropoutRate;
    dropoutRate = hyperParameters.dropoutRate;

    return;
}

HyperParameters HyperParameters::GetHyperParameters()
{
    HyperParameters hyperParameters;

    hyperParameters.learningRate = learningRate;
    hyperParameters.batchSize = batchSize;
    hyperParameters.epochs = epochs;
    hyperParameters.weightDecay = weightDecay;
    hyperParameters.useDropoutRate = useDropoutRate;
    hyperParameters.dropoutRate = dropoutRate;

    return hyperParameters;
}
