#include "Cost.h"

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