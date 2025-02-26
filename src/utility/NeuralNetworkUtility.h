﻿#pragma once
#include <xstring>

class NeuralNetworkUtility
{
public:
    static std::string GetInitTimestamp();
    static std::string GetTimeStampWithAnnotations();
    static std::string FormatTimeHMS(double secondsTotal);
};
