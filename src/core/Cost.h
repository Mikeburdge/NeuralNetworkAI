#pragma once
#include <vector>

enum CostType
{
	meanSquaredError,
	crossEntropy,
	cost_Count
};

class Cost {
public:

	Cost() = default;

	// TODO: Finish implementing this. Must make functionality for including a target variable
	static double CalculateCost(const CostType inCostType, const std::vector<double>& predicted,
	                            const std::vector<double>& actual);

	/// <summary>
	/// Cost calculation function: Mean Squared Error
	/// </summary>
	/// <param name="predicted">Predicted output.</param>
	/// <param name="actual">Actual output.</param>
	/// <returns>Calculated mean squared error.</returns>
	static double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& actual);

	/// <summary>
	/// Cost calculation function: Cross Entropy
	/// </summary>
	/// <param name="predicted">Predicted output.</param>
	/// <param name="actual">Actual output.</param>
	/// <returns>Calculated cross entropy.</returns>
	static double crossEntropy(const std::vector<double>& predicted, const std::vector<double>& actual);

	static std::vector<double> CalculateCostDerivative(CostType inCostType, const std::vector<double>& predicted, const std::vector<double>& actual);
};
