#include "Activation.h"

#include "cmath"
#include "logging/Logger.h"

double Activation::CalculateActivation(const ActivationType inActivation, const double x)
{
    switch (inActivation)
    {
    case ActivationType::sigmoid:
        return sigmoid(x);
    case ActivationType::ReLU:
        return ReLU(x);
    case ActivationType::softmax:
        __debugbreak();
        LOG(LogLevel::ERROR, "Should Never Hapen");
    default:
        return 1.0; // Default return identity for no activation
    }
}

double Activation::CalculateActivationDerivative(const ActivationType inActivation, const double x)
{
    switch (inActivation)
    {
    case ActivationType::sigmoid:
        return sigmoidDerivative(x);
    case ActivationType::ReLU:
        return ReLUDerivative(x);
    case ActivationType::softmax:

    default:
        return 1.0; // Default return identity for no activation
    }
}


double Activation::sigmoid(const double x)
{
    return 1 / (1 + exp(-x));
}

double Activation::sigmoidDerivative(const double x)
{
    //const double sigmoid_x = sigmoid(x);
    // This is post activation. i.e. We have already calculated the sigmoid value and passed that into this function.  
    return x * (1.0 - x);
}

double Activation::ReLU(const double x)
{
    return x > 0 ? x : 0;
}

double Activation::ReLUDerivative(double x)
{
    return x > 0 ? 1.0 : 0;
}
