#pragma once

#include "HyperParameters.h"

#include "Activation.h"

int HyperParameters::defaultInputLayerSize = 784;
int HyperParameters::defaultNumHiddenLayers = 2;
int HyperParameters::defaultHiddenLayerSize = 256; // 128 / 256
int HyperParameters::defaultOutputLayerSize = 10;

float HyperParameters::defaultLearningRate = 0.001f;
int HyperParameters::defaultBatchSize = 64;
int HyperParameters::defaultEpochs = 20;

double HyperParameters::defaultWeightDecay = 0.0005;
bool HyperParameters::defaultUseDropoutRate = true;
float HyperParameters::defaultDropoutRate = 0.2f;
bool HyperParameters::defaultUseGradientClipping = true;
double HyperParameters::defaultGradientClipThreshold = 1.0;

float HyperParameters::learningRate = defaultLearningRate;
int HyperParameters::batchSize = defaultBatchSize;
int HyperParameters::epochs = defaultEpochs;

double HyperParameters::weightDecay = defaultWeightDecay;
bool HyperParameters::useDropoutRate = defaultUseDropoutRate;
float HyperParameters::dropoutRate = defaultDropoutRate;

bool HyperParameters::useGradientClipping = defaultUseGradientClipping;
double HyperParameters::gradientClipThreshold = defaultGradientClipThreshold;

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
