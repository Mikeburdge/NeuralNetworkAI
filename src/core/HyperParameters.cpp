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
bool HyperParameters::defaultUseDropoutRate = false;
float HyperParameters::defaultDropoutRate = 0.3f;

float HyperParameters::learningRate = defaultLearningRate;
int HyperParameters::batchSize = defaultBatchSize;
int HyperParameters::epochs = defaultEpochs;

double HyperParameters::momentum = defaultMomentum;
double HyperParameters::weightDecay = defaultWeightDecay;
bool HyperParameters::useDropoutRate = defaultUseDropoutRate;
float HyperParameters::dropoutRate = defaultDropoutRate;

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
