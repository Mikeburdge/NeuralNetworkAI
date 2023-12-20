#include "Cost.h"

double Cost::CalculateCost(const CostType inCost, const double x)
{
	switch (inCost)
	{
	case CostType::meanSquaredError:

		break;
	case CostType::crossEntropy:
		break;
	case cost_Count:
		break;
	default: ;
	}

	return x;
}

double Cost::meanSquaredError(std::vector<double> predicted, std::vector<double> actual)
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


double Cost::crossEntropy(std::vector<double> predicted, std::vector<double> actual)
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