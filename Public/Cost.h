#pragma once
#include <vector>

class Cost {
public:

	Cost() = default;

	/// <summary>
	/// Cost calculation function: Mean Squared Error
	/// </summary>
	/// <param name="predicted">Predicted output.</param>
	/// <param name="actual">Actual output.</param>
	/// <returns>Calculated mean squared error.</returns>
	static double meanSquaredError(std::vector<double> predicted, std::vector<double> actual);

	/// <summary>
	/// Cost calculation function: Cross Entropy
	/// </summary>
	/// <param name="predicted">Predicted output.</param>
	/// <param name="actual">Actual output.</param>
	/// <returns>Calculated cross entropy.</returns>
	static double crossEntropy(std::vector<double> predicted, std::vector<double> actual);

};
