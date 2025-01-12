#include "Cost.h"

#include "HyperParameters.h"

double Cost::CalculateCost(const CostType inCostType, const std::vector<double>& predicted,
                           const std::vector<double>& actual)
{
	switch (inCostType)
	{
	case CostType::meanSquaredError:
		return meanSquaredError(predicted, actual);
	case CostType::crossEntropy:
		return crossEntropy(predicted, actual);
	case cost_Count:
	default:
		break;
	}

	return 0.0;
}

double Cost::meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& actual)
{
	if (predicted.size() != actual.size()) {
		// Handle error - Input sizes should match
		return -1.0; // Placeholder for error value
	}
	double error = 0.0;
	for (size_t i = 0; i < predicted.size(); ++i) {
		error += pow(predicted[i] - actual[i], 2); // Calculate squared error for each element
	}
	return error / predicted.size(); // Calculate mean squared error
}


double Cost::crossEntropy(const std::vector<double>& predicted, const std::vector<double>& actual)
{
	if (predicted.size() != actual.size()) {
		// Handle error - Input sizes should match
		return -1.0; // Placeholder for error value
	}
	double error = 0.0;
	for (size_t i = 0; i < predicted.size(); ++i) {
		error += actual[i] * log(predicted[i]) + (1 - actual[i]) * log(1 - predicted[i]); // Calculate cross-entropy
	}
	return -error / predicted.size(); // Return negative of the calculated cross-entropy
}

std::vector<double> Cost::CalculateCostDerivative(const CostType inCostType, const std::vector<double>& predicted, const std::vector<double>& actual)
{
	std::vector<double> costGradient(predicted.size());

	for (size_t i = 0; i < predicted.size(); ++i)
	{
		switch (inCostType) {
		case CostType::meanSquaredError:
			// MSE Derivative: predicted - actual
			costGradient[i] = predicted[i] - actual[i];
			break;
		case CostType::crossEntropy:
			// Cross-Entropy Derivative: assuming sigmoid output layer
			costGradient[i] = predicted[i] - actual[i];
			// If I'm going to use softmax then I'll need to use a different approach here. 
			break;
		case cost_Count:
			costGradient[i] = 0.0;
			break;
		}
	}
		
	return costGradient;
}
