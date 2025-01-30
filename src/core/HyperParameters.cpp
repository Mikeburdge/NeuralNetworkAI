#pragma once

#include "HyperParameters.h"

#include "Activation.h"

int HyperParameters::defaultInputLayerSize = 784;
int HyperParameters::defaultNumHiddenLayers = 2;

int HyperParameters::defaultHiddenLayerSize = 256; // 128 / 256
int HyperParameters::defaultOutputLayerSize = 10;

float HyperParameters::defaultLearningRate = 0.004f;
int HyperParameters::defaultBatchSize = 64;
int HyperParameters::defaultEpochs = 10000;

double HyperParameters::defaultMomentum = 0.9;
double HyperParameters::defaultWeightDecay = 0.0000;
bool HyperParameters::defaultUseDropoutRate = true;
float HyperParameters::defaultDropoutRate = 0.3f;
double HyperParameters::defaultGradientClipThreshold = 1.0;
bool HyperParameters::defaultUseGradientClipping = true;

float HyperParameters::learningRate = defaultLearningRate;
int HyperParameters::batchSize = defaultBatchSize;
int HyperParameters::epochs = defaultEpochs;

double HyperParameters::momentum = defaultMomentum;
double HyperParameters::weightDecay = defaultWeightDecay;
bool HyperParameters::useDropoutRate = defaultUseDropoutRate;
float HyperParameters::dropoutRate = defaultDropoutRate;

bool HyperParameters::useGradientClipping = defaultUseGradientClipping;
double HyperParameters::gradientClipThreshold = defaultGradientClipThreshold;

CostType HyperParameters::cost = cost_Count;
ActivationType HyperParameters::activationType = Activation_Count;

int HyperParameters::visualizationUpdateInterval = 5;

void HyperParameters::ResetHyperParameters()
{
	learningRate = defaultLearningRate;
	batchSize = defaultBatchSize; 
	epochs = defaultEpochs; 
    
	momentum = defaultMomentum; 
	weightDecay = defaultWeightDecay;
	useDropoutRate = defaultUseDropoutRate;
	dropoutRate = defaultDropoutRate;
}

void HyperParameters::SetHyperParameters(HyperParameters hyperParameters)
{
	learningRate = hyperParameters.learningRate;
	batchSize = hyperParameters.batchSize;
	epochs = hyperParameters.epochs;
	momentum = hyperParameters.momentum;
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
	hyperParameters.momentum = momentum;
	hyperParameters.weightDecay = weightDecay;
	hyperParameters.useDropoutRate = useDropoutRate;
	hyperParameters.dropoutRate = dropoutRate;
	
	return hyperParameters;
}
