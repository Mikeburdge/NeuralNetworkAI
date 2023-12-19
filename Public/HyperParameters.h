#pragma once

class HyperParameters {
public:

	static int defaultInputLayerSize;
	static int defaultNumHiddenLayers;
	static int defaultHiddenLayerSize;
	static int defaultOutputLayerSize;

	static float defaultLearningRate;
	static int defaultBatchSize;
	static int defaultEpochs;

	static double defaultMomentum;
	static double defaultWeightDecay;
	static bool defaultUseDropoutRate;
	static float defaultDropoutRate;

	static float learningRate; // Learning rate for training.
	static int batchSize; // Size of batches during training.
	static int epochs; // Number of epochs for training.

	static double momentum; // Momentum for optimization (if using momentum-based optimizers).
	static double weightDecay; // Strength of weight decay (if using L2 regularization).
	static bool useDropoutRate;
	static float dropoutRate; // Dropout rate (if implementing dropout regularization).

	static void ResetHyperParameters();


};
