#pragma once

enum ActivationType
{
	sigmoid,
	sigmoidDerivative,
	ReLU,
	Activation_Count
};

class Activation {
public:
	static double CalculateActivation(ActivationType inActivation, double x);

private:

	/// <summary>
	/// Activation function: Sigmoid
	/// </summary>
	/// <param name="x">Input value.</param>
	/// <returns>Output after applying the sigmoid function.</returns>
	static double sigmoid(double x);

	/// <summary>
	/// Calculates the derivative of the sigmoid activation function.
	/// </summary>
	/// <param name="x">Input value.</param>
	/// <returns>Calculated derivative of the sigmoid function.</returns>
	static double sigmoidDerivative(double x);


	/// <summary>
	/// Activation function: ReLU (Rectified Linear Unit)
	/// </summary>
	/// <param name="x">Input value.</param>
	/// <returns>Output after applying the ReLU function.</returns>
	static double ReLU(double x);
};
