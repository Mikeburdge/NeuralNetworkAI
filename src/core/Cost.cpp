#include "Cost.h"
#include <cmath>
#include <vector>

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
    if (predicted.size() != actual.size())
    {
        // Handle error - Input sizes should match
        return -1.0; // Placeholder for error value
    }
    double error = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i)
    {
        error += pow(predicted[i] - actual[i], 2); // Calculate squared error for each element
    }
    return error / predicted.size(); // Calculate mean squared error
}


double Cost::crossEntropy(const std::vector<double>& predicted, const std::vector<double>& actual)
{
    if (predicted.size() != actual.size())
    {
        // Handle error - Input sizes should match
        return -1.0; // Placeholder for error value
    }
    double error = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i)
    {
        // clamp predicted[i] to avoid log(0)
        double p = std::max(1e-15, std::min(1.0 - 1e-15, predicted[i]));
        error += actual[i] * std::log(p) + (1.0 - actual[i]) * std::log(1.0 - p);
    }
    return -error / predicted.size(); // Return negative of the calculated cross-entropy
}

std::vector<double> Cost::CalculateCostDerivative(const CostType inCostType, const std::vector<double>& predicted, const std::vector<double>& actual)
{
    std::vector<double> costGradient(predicted.size());

    switch (inCostType)
    {
    case CostType::meanSquaredError:
    case CostType::crossEntropy:
        // MSE Derivative: predicted - actual
        // Cross-Entropy Derivative: assuming sigmoid output layer
        for (size_t i = 0; i < predicted.size(); ++i)
        {
            costGradient[i] = (predicted[i] - actual[i]);
        }
        break;
    default:
        break;
    }

    return costGradient;
}
