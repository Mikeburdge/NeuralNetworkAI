#pragma once
class Neuron
{
public:
    Neuron() : ActivationValue(0)
    {
    }

    explicit Neuron(const double inActivationValue): ActivationValue(inActivationValue)
    {
    }

    double ActivationValue;
};
