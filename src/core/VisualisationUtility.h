#pragma once
#include <imgui.h>

#include "Activation.h"

class VisualisationUtility
{
public:
    static ImVec4 GetActivationColour(float activationValue, ActivationType activationFunction)
    {
        switch (activationFunction)
        {
        case sigmoid:
            return activationValue > 0.5f ? ImVec4{0.0f, 1.0f, 0.0f, activationValue} : ImVec4{1.0f, 0.0f, 0.0f, 1.0f - activationValue};
        case ReLU:
            return activationValue > 0.0f ? ImVec4{0.0f, 1.0f, 0.0f, activationValue} : ImVec4{1.0f, 0.0f, 0.0f, 1.0f};
        case Activation_Count:
        default:
            return {0.5f, 0.5f, 0.5f, 1.0f};
        }
    }

    static ImVec4 GetWeightColor(float weight)
    {
        float maxIntensity = 1.0f;
        float normalizedWeight = std::min(maxIntensity, fabs(weight) / 5.0f); // Normalizing factor

        if (weight > 0)
        {
            return ImVec4(0.0f, normalizedWeight, 0.0f, 1.0f); // Green for positive weights
        }
        if (weight < 0)
        {
            return ImVec4(normalizedWeight, 0.0f, 0.0f, 1.0f); // Red for negative weights
        }

        return ImVec4(0.5f, 0.5f, 0.5f, 1.0f); // Neutral gray for zero weights
    }

    static void DrawWeightText(ImVec2 lineStart, ImVec2 lineEnd, float weight)
    {
        ImVec2 midPoint = ImVec2((lineStart.x + lineEnd.x) / 2.0f, (lineStart.y + lineEnd.y) / 2.0f);

        float offsetX = 0.0f;
        float offsetY = 0.0f;

        char weightText[16];

        snprintf(weightText, sizeof(weightText), "%.2f", weight);
        ImGui::GetWindowDrawList()->AddText(ImVec2(midPoint.x + offsetX, midPoint.y + offsetY),
                                            IM_COL32(255, 255, 255, 255), weightText);
    }

    static float DistanceToLineSegment(const ImVec2& point, const ImVec2& lineStart, const ImVec2& lineEnd)
    {
        float lineStartSqr(std::pow(lineEnd.x - lineStart.x, 2) + std::pow(lineEnd.y - lineStart.y, 2));

        if (lineStartSqr == 0.0f)
        {
            return std::sqrt(std::pow(point.x - lineStart.x, 2) + std::pow(point.y - lineStart.y, 2));
        }

        float t = ((point.x - lineStart.x) * (lineEnd.x - lineStart.x) + (point.y - lineStart.y) * (lineEnd.y -
            lineStart.y)) / lineStartSqr;
        t = std::max(0.0f, std::min(1.0f, t));

        ImVec2 closestPoint = ImVec2(lineStart.x + t * (lineEnd.x - lineStart.x), lineStart.y + t * (lineEnd.y - lineStart.y));

        return std::sqrt(std::pow(point.x - closestPoint.x, 2) + std::pow(point.y - closestPoint.y, 2));
    }

    static bool PointToCircleCollisionCheck(const ImVec2& circleCenter, const ImVec2& point, const float radius)
    {
        const float distance = (circleCenter.x - point.x) * (circleCenter.x - point.x) + (circleCenter.y - point.y) * (circleCenter.y - point.y);

        return distance < radius * radius;
    }
};
