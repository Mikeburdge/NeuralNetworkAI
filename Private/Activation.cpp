#include "Activation.h"

#include "cmath"

double Activation::CalculateActivation(const ActivationType inActivation, const double x)
{
	switch (inActivation)
	{
	case ActivationType::sigmoid:
		return sigmoid(x);

	case ActivationType::sigmoidDerivative:
		return sigmoidDerivative(x);
	case ActivationType::ReLU:
		return ReLU(x);
	}

	return x;
}

double Activation::sigmoid(const double x) {
	return 1 / (1 + exp(-x));
}

double Activation::sigmoidDerivative(const double x) {
	const double sigmoid_x = sigmoid(x);
	return sigmoid_x * (1 - sigmoid_x);
}

double Activation::ReLU(const double x)
{
	return x > 0 ? x : 0;
}
