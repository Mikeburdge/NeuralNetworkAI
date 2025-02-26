#include "cstdio"          // printf, fprintf
#include "cstdlib"         // abort
#include "imgui_internal.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <filesystem>
#include <stb_image.h>
#include <string>
#include <GLFW/glfw3.h>
#include <ImGuiFileDialog/ImGuiFileDialog.h>
#include <vulkan/vulkan.h>

#include "imgui_internal.h"
#include "core/HyperParameters.h"
#include "core/NeuralNetwork.h"
#include "core/VisualisationUtility.h"
#include "dataloader/MNISTDataSet.h"
#include "implot/implot.h"
#include "logging/Logger.h"
#include "subsystems/NeuralNetworkSubsystem.h"
#include "utility/NeuralNetworkUtility.h"

#include "json.hpp"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#ifdef _DEBUG
#define IMGUI_VULKAN_DEBUG_REPORT
#endif

//--------------------------------------------------------------------
// [SECTION 1] Global Vulkan Data/Variables
//--------------------------------------------------------------------
static VkAllocationCallbacks* g_Allocator = nullptr;
static VkInstance g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_Device = VK_NULL_HANDLE;
static uint32_t g_QueueFamily = (uint32_t)-1;
static VkQueue g_Queue = VK_NULL_HANDLE;
static VkDebugReportCallbackEXT g_DebugReport = VK_NULL_HANDLE;
static VkPipelineCache g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static int g_MinImageCount = 2;
static bool g_SwapChainRebuild = false;

//--------------------------------------------------------------------
// [SECTION 2] Global UI Toggles/State
//--------------------------------------------------------------------
bool showDatasetManagementWindow = true;
bool showNeuralNetworkControlsWindow = true;
bool showVisualizationPanelWindow = false;
bool showAdvancedEditingWindow = false;

// Visualization parameters
float minCircleSizeValue = 5.0f;
float maxCircleSizeValue = 50.0f;
float circleThicknessValue = 1.0f;
float minLineThicknessValue = 1.0f;
bool drawLineConnections = true;
bool drawWeights = false;
bool activeNeuronCanPulse = false;
int hoveredWeightIndex = -1;
int clickedWeightIndex = -1;

// Padding and labeling
float topPadding = 30.0f;
float bottomPadding = 30.0f;
float leftPadding = 30.0f;
float rightPadding = 30.0f;
bool showLayerLabels = true;

// Legend + Training Metrics
bool showLegendWindow = true;
bool showTrainingMetricsWindow = true;
bool showSimpleGraphWindow = false;
bool showAdvancedGraphWindow = true;

std::string filePathName;
std::string filePath;

// Circle Colour stuff
enum class CircleColourMode
{
    DefaultActivation,
    Gradient
};

static CircleColourMode g_CircleColourMode = CircleColourMode::Gradient;

//--------------------------------------------------------------------
// Forward refs
//--------------------------------------------------------------------
static void glfw_error_callback(int error, const char* description);
static void check_vk_result(VkResult err);

//--------------------------------------------------------------------
// A possible texture struct (unused in this snippet, keep if needed)
//--------------------------------------------------------------------
struct MyTextureData
{
    VkDescriptorSet DS;
    int Width;
    int Height;
    int Channels;
    VkImageView ImageView;
    VkImage Image;
    VkDeviceMemory ImageMemory;
    VkSampler Sampler;
    VkBuffer UploadBuffer;
    VkDeviceMemory UploadBufferMemory;
    MyTextureData() { memset(this, 0, sizeof(*this)); }
};

//--------------------------------------------------------------------
// [SECTION 4] Vulkan Helpers
//--------------------------------------------------------------------
uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(g_PhysicalDevice, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    return 0xFFFFFFFF;
}

static void RemoveTexture(MyTextureData* tex_data)
{
    vkFreeMemory(g_Device, tex_data->UploadBufferMemory, nullptr);
    vkDestroyBuffer(g_Device, tex_data->UploadBuffer, nullptr);
    vkDestroySampler(g_Device, tex_data->Sampler, nullptr);
    vkDestroyImageView(g_Device, tex_data->ImageView, nullptr);
    vkDestroyImage(g_Device, tex_data->Image, nullptr);
    vkFreeMemory(g_Device, tex_data->ImageMemory, nullptr);
    ImGui_ImplVulkan_RemoveTexture(tex_data->DS);
}

//--------------------------------------------------------------------
// [SECTION 5] ImGui Windows
//--------------------------------------------------------------------

// 5.1: Advanced Editing (Weights & Biases)
void AdvancedEditingWindow(bool* p_open, NeuralNetwork& network)
{
    if (!ImGui::Begin("Advanced Editing (Weights & Biases)", p_open))
    {
        ImGui::End();
        return;
    }

    for (size_t layerIndex = 0; layerIndex < network.layers.size(); ++layerIndex)
    {
        const Layer& layer = network.layers[layerIndex];
        ImGui::Text("Layer %zu", layerIndex);

        // Biases
        ImGui::Text("Biases:");
        for (size_t neuronIndex = 0; neuronIndex < layer.biases.size(); ++neuronIndex)
        {
            float biasVal = static_cast<float>(layer.biases[neuronIndex]);
            ImGui::SliderFloat(
                ("Bias##" + std::to_string(layerIndex) + "_" + std::to_string(neuronIndex)).c_str(),
                &biasVal, -2.0f, 2.0f, "%.2f"
            );
            network.layers[layerIndex].biases[neuronIndex] = biasVal;
        }

        // Weights
        ImGui::Text("Weights:");
        for (size_t nIndex = 0; nIndex < layer.weights.size(); ++nIndex)
        {
            for (size_t wIndex = 0; wIndex < layer.weights[nIndex].size(); ++wIndex)
            {
                float weightVal = static_cast<float>(layer.weights[nIndex][wIndex]);
                ImGui::SliderFloat(
                    ("Weight##" + std::to_string(layerIndex) + "_" +
                        std::to_string(nIndex) + "_" +
                        std::to_string(wIndex)).c_str(),
                    &weightVal, -5.0f, 5.0f, "%.2f"
                );
                network.layers[layerIndex].weights[nIndex][wIndex] = weightVal;
            }
        }
        ImGui::Separator();
    }

    ImGui::End();
}

// 5.2: Legend Window
void ShowLegendWindow(bool* p_open)
{
    if (!ImGui::Begin("Legend", p_open))
    {
        ImGui::End();
        return;
    }

    ImGui::Text("Neuron Colors:");
    ImGui::BulletText("Green if activation > 0.5, Red otherwise");
    ImGui::Separator();

    ImGui::Text("Line Colors:");
    ImGui::BulletText("Green for positive weights, Red for negative");
    ImGui::BulletText("Thickness indicates magnitude");
    ImGui::Separator();

    ImGui::Text("Output Layer Visualization:");
    ImGui::BulletText("The largest output neuron is scaled to 1.0, \n others are scaled proportionally.");
    ImGui::BulletText("So it's easier to see which neuron is winning.");
    ImGui::Separator();

    ImGui::End();
}

// 5.2.1: Training Metrics Window
void ShowTrainingMetrics(bool* p_open)
{
    if (!ImGui::Begin("Training Metrics", p_open))
    {
        ImGui::End();
        return;
    }

    NeuralNetworkSubsystem& subsystem = NeuralNetworkSubsystem::GetInstance();

    float loss = subsystem.currentLossAtomic.load();
    float rollingAccuracy = subsystem.rollingAccuracyAtomic.load();
    float totalAccuracy = subsystem.currentAccuracyAtomic.load();
    int currentEpoch = subsystem.currentEpochAtomic.load();
    int totalEpochs = subsystem.totalEpochsAtomic.load();

    int currentBatch = subsystem.currentBatchIndexAtomic.load();
    int totalBatchesThisEpoch = subsystem.totalBatchesInEpochAtomic.load();

    int currentBatchSize = subsystem.currentBatchSizeAtomic.load();
    int correctPredictionsThisBatch = subsystem.correctPredictionsThisBatchAtomic.load();
    int totalCorrectPredictions = subsystem.totalCorrectPredictionsAtomic.load();
    int totalPredictions = subsystem.totalPredictionsAtomic.load();

    double batchTime = subsystem.totalBatchTimeAtomic.load();
    double averageBatchTime = subsystem.averageBatchTimeAtomic.load();
    double samplesPerSecond = subsystem.samplesPerSecAtomic.load();


    float epochFraction = 0.0f;
    if (totalBatchesThisEpoch > 0)
    {
        epochFraction = static_cast<float>(currentBatch) / static_cast<float>(totalBatchesThisEpoch);
    }

    float overallFraction = 0.0f;
    if (totalEpochs > 0)
    {
        overallFraction = (static_cast<float>(currentEpoch) + epochFraction) / static_cast<float>(totalEpochs);
    }

    ImGui::ProgressBar(epochFraction, ImVec2(-1.0f, 0.0f), "Epoch Progress");
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("%.1f%% done", epochFraction * 100.0f);
    }
    ImGui::ProgressBar(overallFraction, ImVec2(-1.0f, 0.0f), "Training Progress");
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("%.1f%% done", overallFraction * 100.0f);
    }

    if (subsystem.trainingTimer.isInitialized)
    {
        std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - subsystem.trainingTimer.startTime).count();

        double epochTime = 0.0;
        if (currentEpoch > 0)
        {
            epochTime = subsystem.trainingTimer.epochDuration;
        }
        else
        {
            if (currentEpoch > 0)
            {
                epochTime = elapsed / static_cast<double>(currentEpoch);
            }
            else
            {
                epochTime = 0.0;
            }
        }

        const int epochsLeft = totalEpochs - (currentEpoch + 1);
        const double estimateRemainingTime = epochsLeft * epochTime;


        ImGui::Text("Training Time Elapsed: %s", NeuralNetworkUtility::FormatTimeHMS(elapsed).c_str());
        ImGui::Text("Training Time Remaining: %s", NeuralNetworkUtility::FormatTimeHMS(estimateRemainingTime).c_str());
    }
    else
    {
        ImGui::Text("Training Time Elapsed: N/A");
        ImGui::Text("Training Time Remaining: N/A");
    }


    ImGui::Text("Epoch: %d/%d (Batch %d/%d)", currentEpoch + 1, totalEpochs, currentBatch, totalBatchesThisEpoch);
    ImGui::Text("Loss: %.4f", loss);
    ImGui::Text("Rolling Accuracy (last 1000): %.2f%%", rollingAccuracy * 100.f);
    ImGui::Text("Total Accuracy: %.2f%%", totalAccuracy * 100.f);

    ImGui::Separator();
    ImGui::Text("Correct Predictions (This Batch): %d / %d", correctPredictionsThisBatch, currentBatchSize);
    ImGui::Text("Correct Predictions (Overall): %d / %d", totalCorrectPredictions, totalPredictions);

    ImGui::Text("Batch Time | Last: %.2fs| Average: %.2fs", batchTime, averageBatchTime);
    ImGui::Text("Samples/sec: %.0f", samplesPerSecond);

    ImGui::End();
}


// 5.2.2: Training Metrics Graph Window

bool showLossGraph = true;
bool showAccuracyGraph = true;
bool showRollingAccuracyGraph = true;

// 5.2.2: Training Metrics Graph Window 


void ShowSimpleGraphWindow(bool* p_open)
{
    if (!ImGui::Begin("Basic ImGui Graph", p_open))
    {
        ImGui::End();
        return;
    }

    NeuralNetworkSubsystem& subsystem = NeuralNetworkSubsystem::GetInstance();

    static std::vector<float> accuracyData;
    accuracyData.clear();

    {
        std::lock_guard<std::mutex> lock(subsystem.metricMutex);

        accuracyData.reserve(subsystem.trainingHistory.size());

        for (auto& pt : subsystem.trainingHistory)
        {
            accuracyData.push_back(pt.accuracy);
        }
    }

    if (!accuracyData.empty())
    {
        ImGui::PlotLines("Accuracy Over Iterations",
                         accuracyData.data(),
                         static_cast<int>(accuracyData.size()),
                         0, // offset
                         nullptr, // optional overlay text
                         FLT_MAX, // min scale
                         FLT_MAX, // max scale
                         ImVec2(0, 100)); // size of the plot in pixels (width=auto, height=100)
    }
    else
    {
        ImGui::Text("No training data yet...");
    }

    ImGui::End();
}

enum class MetricDisplayType
{
    Percentage,
    Decimal,
    Unknown
};

MetricDisplayType GetMetricDisplayType(const char* metricName)
{
    if (strcmp(metricName, "Loss") == 0) return MetricDisplayType::Decimal;
    if (strcmp(metricName, "Accuracy") == 0) return MetricDisplayType::Percentage;
    if (strcmp(metricName, "Roll") == 0) return MetricDisplayType::Percentage;
    return MetricDisplayType::Unknown;
}

void ShowAdvancedGraphWindow(bool* p_open)
{
    // 1) Create the ImGui Window
    if (!ImGui::Begin("Advanced Graph Window (Time + Hover)", p_open))
    {
        ImGui::End();
        return;
    }

    // Access the singleton subsystem
    NeuralNetworkSubsystem& subsystem = NeuralNetworkSubsystem::GetInstance();

    // 2) Toggles
    static bool showLoss = true;
    static bool showAccuracy = true;
    static bool showRollingAcc = true;

    // UI checkboxes
    ImGui::Checkbox("Show Loss", &showLoss);
    ImGui::SameLine();
    ImGui::Checkbox("Show Accuracy", &showAccuracy);
    ImGui::SameLine();
    ImGui::Checkbox("Show RollingAcc", &showRollingAcc);
    ImGui::Separator();

    // 3) Gather data from trainingHistory
    static std::vector<float> timeData;
    static std::vector<float> lossData;
    static std::vector<float> accData;
    static std::vector<float> rollingData;

    // Clear old data
    timeData.clear();
    lossData.clear();
    accData.clear();
    rollingData.clear();

    // Lock because trainingHistory is shared
    {
        std::lock_guard<std::mutex> lock(subsystem.metricMutex);

        size_t count = subsystem.trainingHistory.size();
        timeData.reserve(count);
        lossData.reserve(count);
        accData.reserve(count);
        rollingData.reserve(count);

        // Copy out
        for (auto& pt : subsystem.trainingHistory)
        {
            timeData.push_back(pt.timeSeconds);
            lossData.push_back(pt.loss);
            accData.push_back(pt.accuracy);
            rollingData.push_back(pt.rollingAcc);
        }
    }

    // If no data, inform & return
    if (timeData.empty())
    {
        ImGui::Text("No training data to display yet.");
        ImGui::End();
        return;
    }

    // 4) Determine min/max Y & min/max Time
    float globalMinY = FLT_MAX;
    float globalMaxY = -FLT_MAX;

    auto UpdateMinMax = [&](const std::vector<float>& data)
    {
        for (float v : data)
        {
            if (v < globalMinY) globalMinY = v;
            if (v > globalMaxY) globalMaxY = v;
        }
    };

    // Update domain only if toggles are on
    if (showLoss && !lossData.empty()) UpdateMinMax(lossData);
    if (showAccuracy && !accData.empty()) UpdateMinMax(accData);
    if (showRollingAcc && !rollingData.empty()) UpdateMinMax(rollingData);

    // If still unchanged => no valid data
    if (globalMinY == FLT_MAX || globalMaxY == -FLT_MAX)
    {
        ImGui::Text("No selected metrics have data.");
        ImGui::End();
        return;
    }

    // Time domain
    float minTime = timeData.front();
    float maxTime = timeData.back();
    // If data is out of order, consider scanning to find the true minTime & maxTime.

    // If everything identical, pad a bit
    if (globalMinY == globalMaxY)
    {
        globalMaxY += 1.0f;
        globalMinY -= 1.0f;
    }
    if (minTime == maxTime)
    {
        maxTime += 1.0f;
        minTime -= 1.0f;
    }

    // 5) Show optional debug info (or remove if not needed)
    ImGui::Text("Max Rolling Accuracy: %.3f", globalMaxY);
    ImGui::Text("Y Range: [%.3f .. %.3f]", globalMinY, globalMaxY);
    ImGui::Separator();

    // 6) Identify metric display types -> Y-axis label
    std::vector<MetricDisplayType> activeTypes;
    if (showLoss) activeTypes.push_back(GetMetricDisplayType("Loss"));
    if (showAccuracy) activeTypes.push_back(GetMetricDisplayType("Accuracy"));
    if (showRollingAcc) activeTypes.push_back(GetMetricDisplayType("Roll"));

    MetricDisplayType finalType = MetricDisplayType::Unknown;
    bool allSameType = true;
    for (size_t i = 0; i < activeTypes.size(); i++)
    {
        if (i == 0) finalType = activeTypes[i];
        else
        {
            if (activeTypes[i] != finalType)
            {
                allSameType = false;
                break;
            }
        }
    }

    std::string yAxisLabel;
    if (!activeTypes.empty())
    {
        if (!allSameType)
        {
            yAxisLabel = "Y Axis (Mixed)";
        }
        else
        {
            switch (finalType)
            {
            case MetricDisplayType::Percentage: yAxisLabel = "Percentage (%)";
                break;
            case MetricDisplayType::Decimal: yAxisLabel = "Value";
                break;
            default: yAxisLabel = "Y Axis";
                break;
            }
        }
    }
    else
    {
        yAxisLabel = "Y Axis";
    }

    // 7) Now the child region for the actual graph
    ImVec2 canvasSize(ImGui::GetContentRegionAvail().x, 300.0f);
    ImGui::BeginChild("TimeGraphCanvas", canvasSize, true /*border*/);

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 size = ImGui::GetContentRegionAvail();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // Background
    ImU32 bgColor = IM_COL32(40, 40, 40, 255);
    drawList->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), bgColor);

    float pad = 55.0f; // bumped up a bit for left labels
    ImVec2 graphStart(pos.x + pad, pos.y + pad);
    ImVec2 graphEnd(pos.x + size.x - pad, pos.y + size.y - pad);

    float width = graphEnd.x - graphStart.x;
    float height = graphEnd.y - graphStart.y;

    // 8) Y-axis label (vertical text)
    if (!yAxisLabel.empty())
    {
        float midY = 0.5f * (graphStart.y + graphEnd.y) - 120.0f;
        ImVec2 textPos(graphStart.x - 30.0f, midY);
        drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                          textPos,
                          IM_COL32_WHITE,
                          yAxisLabel.c_str(),
                          NULL,
                          0.0f
        );
    }

    // 9) Function to transform (time, val) -> ImVec2
    auto transform = [&](float t, float val)
    {
        float xFrac = (t - minTime) / (maxTime - minTime);
        float yFrac = (val - globalMinY) / (globalMaxY - globalMinY);
        float xPos = graphStart.x + (xFrac * width);
        float yPos = graphEnd.y - (yFrac * height);
        return ImVec2(xPos, yPos);
    };

    // 10) Y-axis ticks & numeric labels
    // Example: 5 or 6 ticks
    int numYTicks = 6;
    for (int i = 0; i < numYTicks; i++)
    {
        float frac = (float)i / (float)(numYTicks - 1);
        float yVal = globalMinY + frac * (globalMaxY - globalMinY);
        float yPix = graphEnd.y - frac * height;

        // A short horizontal line to mark the tick
        drawList->AddLine(ImVec2(graphStart.x, yPix),
                          ImVec2(graphStart.x - 5, yPix),
                          IM_COL32_WHITE);

        // Y label text
        char buf[32];
        // e.g. 2 decimals
        snprintf(buf, sizeof(buf), "%.2f", yVal);

        ImVec2 textSize = ImGui::CalcTextSize(buf);
        // Put the text a bit left of the tick
        float textX = graphStart.x - textSize.x - 8.0f;
        float textY = yPix - textSize.y * 0.5f;
        drawList->AddText(ImVec2(textX, textY), IM_COL32_WHITE, buf);
    }

    // 11) helper to draw a single metric line
    auto drawMetricLine = [&](const std::vector<float>& dataVec, ImU32 color)
    {
        int count = (int)dataVec.size();
        for (int i = 0; i < count - 1; i++)
        {
            float t1 = timeData[i], t2 = timeData[i + 1];
            float v1 = dataVec[i], v2 = dataVec[i + 1];
            ImVec2 p1 = transform(t1, v1);
            ImVec2 p2 = transform(t2, v2);
            drawList->AddLine(p1, p2, color, 2.0f);
        }
    };

    // 12) Draw lines for toggled metrics
    if (showLoss && !lossData.empty())
        drawMetricLine(lossData, IM_COL32(255, 165, 0, 255)); // Orange
    if (showAccuracy && !accData.empty())
        drawMetricLine(accData, IM_COL32(0, 255, 0, 255)); // Green
    if (showRollingAcc && !rollingData.empty())
        drawMetricLine(rollingData, IM_COL32(255, 105, 180, 255)); // HotPink

    // 13) X-axis Ticks
    int numXTicks = 6;
    for (int i = 0; i < numXTicks; i++)
    {
        float tickFrac = (float)i / (float)(numXTicks - 1);
        float tickTime = minTime + tickFrac * (maxTime - minTime);

        float tickX = graphStart.x + tickFrac * width;
        // small vertical line for the tick
        drawList->AddLine(ImVec2(tickX, graphEnd.y),
                          ImVec2(tickX, graphEnd.y + 5),
                          IM_COL32_WHITE);

        // label
        char lbl[64];
        snprintf(lbl, sizeof(lbl), "%s", NeuralNetworkUtility::FormatTimeHMS(tickTime).c_str());
        ImVec2 textSize = ImGui::CalcTextSize(lbl);
        float tx = tickX - textSize.x * 0.5f;
        float ty = graphEnd.y + 6;
        drawList->AddText(ImVec2(tx, ty), IM_COL32_WHITE, lbl);
    }

    // 14) Mouse hover => vertical line & tooltip
    ImVec2 canvasMin = graphStart;
    ImVec2 canvasMax = graphEnd;
    bool hovered = ImGui::IsMouseHoveringRect(canvasMin, canvasMax);
    if (hovered)
    {
        ImVec2 mousePosInWindow = ImGui::GetMousePos();
        if (mousePosInWindow.x < canvasMin.x) mousePosInWindow.x = canvasMin.x;
        if (mousePosInWindow.x > canvasMax.x) mousePosInWindow.x = canvasMax.x;

        float xFrac = (mousePosInWindow.x - canvasMin.x) / (canvasMax.x - canvasMin.x);
        float hoveredTime = minTime + xFrac * (maxTime - minTime);

        // Find the closest data index
        int closestIndex = 0;
        float minDist = FLT_MAX;
        for (int i = 0; i < (int)timeData.size(); i++)
        {
            float dist = fabsf(timeData[i] - hoveredTime);
            if (dist < minDist)
            {
                minDist = dist;
                closestIndex = i;
            }
        }

        // Draw the vertical line
        float lineX = transform(timeData[closestIndex], 0).x;
        drawList->AddLine(ImVec2(lineX, canvasMin.y),
                          ImVec2(lineX, canvasMax.y),
                          IM_COL32(255, 255, 255, 120),
                          1.0f);

        // Tooltip with the values
        if (ImGui::IsMouseHoveringRect(canvasMin, canvasMax))
        {
            float tVal = timeData[closestIndex];
            float lVal = (showLoss) ? lossData[closestIndex] : NAN;
            float aVal = (showAccuracy) ? accData[closestIndex] : NAN;
            float rVal = (showRollingAcc) ? rollingData[closestIndex] : NAN;

            ImGui::BeginTooltip();
            ImGui::Text("Time: %s", NeuralNetworkUtility::FormatTimeHMS(tVal).c_str());
            if (showLoss) ImGui::Text("Loss = %.3f", lVal);
            if (showAccuracy) ImGui::Text("Acc  = %.2f%%", aVal);
            if (showRollingAcc) ImGui::Text("Roll = %.2f%%", rVal);
            ImGui::EndTooltip();
        }
    }

    // End child region
    ImGui::EndChild();

    // End main window
    ImGui::End();
}


// 5.3: Calculation for circle size
static float CalculateMaxCircleSize(const ImVec2& winSize, int numLayers, float maxCircleSizeValue)
{
    // same as before
    return std::min(std::min(winSize.x, winSize.y) / (numLayers * 2.0f), maxCircleSizeValue);
}

// 5.4: Visualization Panel
void VisualizationPanelWindow(bool* p_open, const NeuralNetwork& network)
{
    // Begin the main visualization window
    if (!ImGui::Begin("Neural Network Visualization (Panel)", p_open))
    {
        ImGui::End();
        return;
    }

    NeuralNetworkSubsystem& nnSubsystem = NeuralNetworkSubsystem::GetInstance();
    // for now im just gonna check the layers but I really should add a bool in the subsystem
    bool bIsNetworkInitialized = !nnSubsystem.GetNeuralNetwork().layers.empty();

    if (!bIsNetworkInitialized)
    {
        ImGui::Text("No Neural Network Initialized");
        ImGui::End();
        return;
    }

    const int layerCount = (int)network.layers.size();
    if (layerCount == 0)
    {
        ImGui::Text("No layers to display.");
        ImGui::End();
        return;
    }

    // Grab the size & position of this window
    ImVec2 availSize = ImGui::GetWindowSize();
    ImVec2 windowPos = ImGui::GetWindowPos();
    float scrollY = ImGui::GetScrollY();

    float innerHeight = availSize.y - (topPadding + bottomPadding);
    float innerWidth = availSize.x - (leftPadding + rightPadding);

    float maxCircSize = CalculateMaxCircleSize(availSize, layerCount, maxCircleSizeValue);

    float layerSpacing = (layerCount > 1) ? (availSize.x / (float)(layerCount + 1)) : (availSize.x * 0.5f);

    struct LineInfo
    {
        ImVec2 lineStart;
        ImVec2 lineEnd;
        float weight;
        bool isHovered;

        LineInfo(const ImVec2& s, const ImVec2& e, float w, bool hov)
            : lineStart(s), lineEnd(e), weight(w), isHovered(hov)
        {
        }
    };
    std::vector<LineInfo> lineInfos;

    int pseudoInputCount = network.layers[0].numNeuronsOutOfPreviousLayer;
    float inputLayerPosX = layerSpacing * 0.5f;
    int maxNeuronDisplay = NeuralNetworkSubsystem::GetInstance().maxNeuronsToDisplay;
    int displayedInputs = std::min(pseudoInputCount, maxNeuronDisplay);

    float inputNeuronSpacing = (displayedInputs > 1) ? (innerHeight / (float)(displayedInputs + 1)) : (innerHeight * 0.5f);

    if (showLayerLabels)
    {
        float labelY = windowPos.y + scrollY + topPadding;
        ImVec2 labelPos(windowPos.x + inputLayerPosX, labelY);
        ImGui::GetWindowDrawList()->AddText(labelPos, IM_COL32(255, 255, 0, 255), "Input Layer");
    }

    std::vector<ImVec2> pseudoInputCenters;
    pseudoInputCenters.reserve(displayedInputs);

    for (int i = 0; i < displayedInputs; ++i)
    {
        float circleSize = std::min(maxCircSize, inputNeuronSpacing * 0.5f);

        float posY = topPadding + (i + 1) * inputNeuronSpacing;

        ImVec2 circleCenter(
            windowPos.x + inputLayerPosX,
            windowPos.y + posY - scrollY
        );

        ImColor circleColor = ImColor(0.7f, 0.7f, 0.7f, 1.0f);

        ImGui::GetWindowDrawList()->AddCircle(
            circleCenter,
            circleSize,
            circleColor,
            16, // segments
            circleThicknessValue
        );

        char label[16];
        std::snprintf(label, sizeof(label), "X%d", i);
        ImVec2 labelSize = ImGui::CalcTextSize(label);
        ImVec2 textPos(
            circleCenter.x - labelSize.x * 0.5f,
            circleCenter.y - labelSize.y * 0.5f
        );
        ImGui::GetWindowDrawList()->AddText(textPos, IM_COL32_WHITE, label);

        pseudoInputCenters.push_back(circleCenter);
    }

    if (pseudoInputCount > displayedInputs)
    {
        float truncatedY = topPadding + displayedInputs * (innerHeight / (displayedInputs + 1));
        ImVec2 truncatedPos(
            windowPos.x + inputLayerPosX - 40.0f,
            windowPos.y + truncatedY + 10.0f - scrollY
        );
        char msg[64];
        std::snprintf(msg, sizeof(msg), "Showing %d of %d", displayedInputs, pseudoInputCount);
        ImGui::GetWindowDrawList()->AddText(truncatedPos, IM_COL32(255, 255, 0, 255), msg);
    }

    for (int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
    {
        const Layer& layer = network.layers[layerIndex];
        int numNeurons = layer.numNeurons;
        int displayCount = std::min(numNeurons, maxNeuronDisplay);

        float neuronSpacing = (displayCount > 1) ? (innerHeight / (float)(displayCount + 1)) : (innerHeight * 0.5f);

        float layerPosX = (layerIndex + 1) * layerSpacing;

        if (showLayerLabels)
        {
            std::string layerName;
            if (layerIndex == layerCount - 1)
                layerName = "Output Layer";
            else
                layerName = "Hidden Layer " + std::to_string(layerIndex);

            float labelY = windowPos.y + scrollY + topPadding;
            ImVec2 labelPos(windowPos.x + layerPosX, labelY);
            ImGui::GetWindowDrawList()->AddText(labelPos, IM_COL32(255, 255, 0, 255), layerName.c_str());
        }

        static std::vector<float> displayActivations;
        displayActivations.resize(displayCount);

        for (int n = 0; n < displayCount; ++n)
        {
            float activationVal = (float)layer.neurons[n].ActivationValue;
            displayActivations[n] = activationVal;
        }

        bool isFinalLayer = (layerIndex == layerCount - 1);
        if (isFinalLayer)
        {
            float maxVal = 0.0f;
            for (int i = 0; i < displayCount; i++)
            {
                if (displayActivations[i] > maxVal) maxVal = displayActivations[i];
            }
            if (maxVal > 0.0f)
            {
                for (int i = 0; i < displayCount; i++)
                    displayActivations[i] /= maxVal;
            }
        }

        // Draw each neuron
        for (int neuronIndex = 0; neuronIndex < displayCount; ++neuronIndex)
        {
            float circleSize = std::min(maxCircSize, neuronSpacing * 0.5f);
            float posY = topPadding + (neuronIndex + 1) * neuronSpacing;
            ImVec2 circleCenter(
                windowPos.x + layerPosX,
                windowPos.y + posY - scrollY
            );

            // activation-based color
            float activationVal = displayActivations[neuronIndex];

            ImColor baseColor;
            switch (g_CircleColourMode)
            {
            case CircleColourMode::DefaultActivation:
                baseColor = VisualisationUtility::GetActivationColour(
                    activationVal, HyperParameters::activationType);
                break;

            case CircleColourMode::Gradient:
            default:
                {
                    float ratio = std::max(0.0f, std::min(1.0f, activationVal));
                    float red = 1.0f - ratio;
                    float green = ratio;
                    baseColor = ImColor(red, green, 0.0f, 1.0f);
                }
                break;
            }

            // Possibly pulse if over 0.5
            float actualSize = circleSize;
            if (activeNeuronCanPulse && activationVal > 0.5f)
            {
                actualSize += sinf(ImGui::GetTime() * 5.0f) * 2.0f;
            }

            // Hover detection
            ImVec2 mousePos = ImGui::GetMousePos();
            float dx = mousePos.x - circleCenter.x;
            float dy = mousePos.y - circleCenter.y;
            bool hoveredNeuron = (dx * dx + dy * dy) < (actualSize * actualSize);

            // neuron text
            char buf[32];
            if (hoveredNeuron)
                std::snprintf(buf, sizeof(buf), "%.3f", activationVal);
            else
                std::snprintf(buf, sizeof(buf), "%.1f", activationVal);

            // highlight if hovered
            ImColor drawColor = hoveredNeuron ? ImColor(255, 255, 0, 255) : baseColor;

            // Draw circle
            ImGui::GetWindowDrawList()->AddCircle(
                circleCenter,
                actualSize,
                drawColor,
                16,
                circleThicknessValue
            );

            // Draw text
            ImVec2 txtSize = ImGui::CalcTextSize(buf);
            ImVec2 txtPos(
                circleCenter.x - txtSize.x * 0.5f,
                circleCenter.y - txtSize.y * 0.5f
            );
            ImGui::GetWindowDrawList()->AddText(txtPos, IM_COL32_WHITE, buf);

            if (drawLineConnections)
            {
                int prevLayerCount = (layerIndex == 0) ? pseudoInputCount : network.layers[layerIndex - 1].numNeurons;

                // limit how many lines we draw
                int prevDisplayCount = std::min(prevLayerCount, maxNeuronDisplay);

                float prevPosX = (layerIndex == 0) ?
                                     inputLayerPosX // from pseudo input
                                     :
                                     ((layerIndex) * layerSpacing);

                float prevNeuronSpacing = (prevDisplayCount > 1) ? (innerHeight / (float)(prevDisplayCount + 1)) : (innerHeight * 0.5f);

                for (int pIdx = 0; pIdx < prevDisplayCount; ++pIdx)
                {
                    float weightVal = layer.weights[neuronIndex][pIdx];

                    float prevPosY = topPadding + (pIdx + 1) * prevNeuronSpacing;
                    ImVec2 lineStart(
                        windowPos.x + prevPosX + circleSize,
                        windowPos.y + prevPosY - scrollY
                    );
                    ImVec2 lineEnd(
                        circleCenter.x - circleSize,
                        circleCenter.y
                    );

                    // hover detection
                    float distLine = VisualisationUtility::DistanceToLineSegment(
                        mousePos, lineStart, lineEnd);
                    bool highlightBecauseNeuron = hoveredNeuron;
                    bool hoveredLine = (distLine < (minLineThicknessValue + 2.0f))
                        || highlightBecauseNeuron;
                    bool clickedLine = (hoveredLine && ImGui::IsMouseClicked(0));

                    if (hoveredLine)
                    {
                        hoveredWeightIndex = neuronIndex + pIdx;
                        if (clickedLine)
                            clickedWeightIndex = hoveredWeightIndex;
                    }

                    float thickness = std::max(minLineThicknessValue,
                                               1.0f + std::min(4.0f, (float)fabs(weightVal) / 2.0f));
                    ImColor lineColor = hoveredLine ? ImColor(255, 255, 0, 255) : ImColor(VisualisationUtility::GetWeightColor(weightVal));

                    ImGui::GetWindowDrawList()->AddLine(
                        lineStart,
                        lineEnd,
                        lineColor,
                        thickness
                    );

                    // If we either hover or have "drawWeights" on, store line info
                    if (drawWeights || hoveredLine)
                    {
                        lineInfos.emplace_back(lineStart, lineEnd, weightVal, hoveredLine);
                    }
                }
            }
        }

        if (numNeurons > displayCount)
        {
            float lastNeuronY = topPadding + displayCount * (innerHeight / (displayCount + 1));
            ImVec2 truncatedPos(
                windowPos.x + layerPosX - 40.0f,
                windowPos.y + lastNeuronY + 10.0f - scrollY
            );
            char msg[64];
            std::snprintf(msg, sizeof(msg), "Showing %d of %d", displayCount, numNeurons);
            ImGui::GetWindowDrawList()->AddText(truncatedPos, IM_COL32(255, 255, 0, 255), msg);
        }
    }

    // Draw numeric weight text if we have lineInfos
    for (auto& info : lineInfos)
    {
        VisualisationUtility::DrawWeightText(info.lineStart, info.lineEnd, info.weight);
    }

    ImGui::End();

    // The separate customization window
    ImGui::Begin("Visualization Customization");
    ImGui::SliderInt("Max Neurons Displayed", &NeuralNetworkSubsystem::GetInstance().maxNeuronsToDisplay, 0, 300);

    ImGui::SliderFloat("Top Padding", &topPadding, 0.0f, 300.0f);
    ImGui::SliderFloat("Bottom Padding", &bottomPadding, 0.0f, 300.0f);
    ImGui::SliderFloat("Left Padding", &leftPadding, 0.0f, 300.0f);
    ImGui::SliderFloat("Right Padding", &rightPadding, 0.0f, 300.0f);

    ImGui::Checkbox("Show Layer Labels", &showLayerLabels);
    ImGui::Checkbox("Draw Lines", &drawLineConnections);
    ImGui::Checkbox("Draw Weights", &drawWeights);
    ImGui::Checkbox("Pulsating Neurons", &activeNeuronCanPulse);

    ImGui::SliderFloat("Min Circle Size", &minCircleSizeValue, 1.0f, 10.0f);
    ImGui::SliderFloat("Max Circle Size", &maxCircleSizeValue, 10.0f, 100.0f);
    ImGui::SliderFloat("Circle Thickness", &circleThicknessValue, 1.0f, 5.0f);
    ImGui::SliderFloat("Min Line Thickness", &minLineThicknessValue, 1.0f, 5.0f);

    ImGui::Checkbox("Show Legend Window", &showLegendWindow);
    ImGui::Checkbox("Show Training Metrics", &showTrainingMetricsWindow);
    ImGui::Checkbox("Show Simple Training Metrics Graph", &showSimpleGraphWindow);
    ImGui::Checkbox("Show Advanced Metrics Graph", &showAdvancedGraphWindow);

    ImGui::Text("Neuron Colour Mode");
    static int colourModeIndex = 0;
    const char* colourModes[] = {"Default", "Red-Green Gradient"};
    if (ImGui::BeginCombo("Colour Mode", colourModes[colourModeIndex]))
    {
        for (int i = 0; i < IM_ARRAYSIZE(colourModes); i++)
        {
            bool isSelected = (colourModeIndex == i);
            if (ImGui::Selectable(colourModes[i], isSelected))
            {
                colourModeIndex = i;
                g_CircleColourMode = (i == 0 ? CircleColourMode::DefaultActivation : CircleColourMode::Gradient);
            }
            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    static int vizInt = 10;
    ImGui::InputInt("Visualization Interval (batches)", &vizInt);
    if (ImGui::Button("Apply Visualization Interval"))
    {
        NeuralNetworkSubsystem::GetInstance().SetVizUpdateInterval(vizInt);
    }

    ImGui::End();

    if (showLegendWindow)
    {
        ShowLegendWindow(&showLegendWindow);
    }
    if (showTrainingMetricsWindow)
    {
        ShowTrainingMetrics(&showTrainingMetricsWindow);
    }
    if (showSimpleGraphWindow)
    {
        ShowSimpleGraphWindow(&showSimpleGraphWindow);
    }
    if (showAdvancedGraphWindow)
    {
        ShowAdvancedGraphWindow(&showAdvancedGraphWindow);
    }
}

// 5.5: Dataset Management
static bool useTextPreview = false;
static char digitText[2] = "0";
static bool bHasLoadedMNISTTestData = false;


static float imageZoomLevel = 2.0f;
static int selectedImageIndex = -1; // To track selected image

void DatasetManagementWindow(bool* p_open, NeuralNetwork& network)
{
    if (!ImGui::Begin("Dataset Management", p_open))
    {
        ImGui::End();
        return;
    }

    ImGui::TextWrapped("This panel lets you load MNIST data and train automatically.");

    bool isDataLoaded = NeuralNetworkSubsystem::GetInstance().IsMNISTTrainingDataLoaded();
    if (isDataLoaded)
    {
        auto size = NeuralNetworkSubsystem::GetInstance().GetTrainingDataSet().Size();
        ImGui::Text("MNIST data is loaded: %zu samples.", size);
    }
    else
    {
        ImGui::Text("MNIST data is not loaded yet.");
    }

    if (NeuralNetworkSubsystem::GetInstance().IsTrainingInProgress())
    {
        if (ImGui::Button("Pause Training"))
        {
            NeuralNetworkSubsystem::GetInstance().StopTraining();
        }
        ImGui::SameLine();

        if (ImGui::Button("Restart Training"))
        {
            // Reset training state
            NeuralNetworkSubsystem::GetInstance().StopTraining();
            NeuralNetworkSubsystem::GetInstance().currentEpochAtomic.store(0);
            {
                std::lock_guard<std::mutex> lock(NeuralNetworkSubsystem::GetInstance().metricMutex);
                NeuralNetworkSubsystem::GetInstance().trainingHistory.clear();
            }
            // Start fresh training
            NeuralNetworkSubsystem::GetInstance().TrainOnMNISTFullProcess();
        }
    }
    else
    {
        if (ImGui::Button("Train MNIST (Full Process)"))
        {
            NeuralNetworkSubsystem& subsystem = NeuralNetworkSubsystem::GetInstance();

            if (subsystem.GetNeuralNetwork().layers.empty())
            {
                LOG(LogLevel::INFO, "No existing network. Auto-creating network with selected values.");

                subsystem.InitNeuralNetwork(
                    ActivationType::sigmoid, // Assuming ActivationType::sigmoid exists
                    CostType::crossEntropy, // Assuming CostType::crossEntropy exists
                    HyperParameters::defaultInputLayerSize,
                    HyperParameters::defaultNumHiddenLayers,
                    HyperParameters::defaultHiddenLayerSize,
                    HyperParameters::defaultOutputLayerSize
                );
            }

            subsystem.SetVisualizationCallback([](const NeuralNetwork& net)
            {
                // Callback function to handle visualization updates if needed
                // For simplicity, left empty
            });

            showDatasetManagementWindow = true;
            showVisualizationPanelWindow = true;

            NeuralNetworkSubsystem::GetInstance().TrainOnMNISTFullProcess();
        }

        ImGui::SameLine();

        if (ImGui::Button("Continue Training"))
        {
            // Continue training asynchronously
            NeuralNetworkSubsystem::GetInstance().TrainOnMNISTAsync();
        }
    }

    if (NeuralNetworkSubsystem::GetInstance().currentEpochAtomic.load() >= HyperParameters::epochs)
    {
        ImGui::Text("All Epochs Completed.");
    }

    ImGui::Separator();

    if (ImGui::Button("Evaluate on Test Set"))
    {
        double acc;
        NeuralNetworkSubsystem::GetInstance().EvaluateTestSet(); // Assuming EvaluateTestSet logs the accuracy
        // Optionally, retrieve and display the accuracy if the function returns it
    }
    ImGui::Separator();

    if (ImGui::Button("Test Custom Set"))
    {
        NeuralNetworkSubsystem::GetInstance().TestCustomSet();
    }

    // Retrieve test set data  
    const auto& testSetImages = NeuralNetworkSubsystem::GetInstance().GetTestSetImages();
    const auto& testSetPredictions = NeuralNetworkSubsystem::GetInstance().GetTestSetPredictions();
    const auto& testSetConfidences = NeuralNetworkSubsystem::GetInstance().GetTestSetConfidence();

    if (!testSetImages.empty() && !testSetPredictions.empty())
    {
        ImGui::Separator();
        ImGui::Text("Custom Test Set Inference Results:");
        ImGui::Separator();

        // Begin a child region for the dual panes  
        ImGui::BeginChild("DualPane", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

        // Determine the size of each pane  
        float paneWidth = ImGui::GetContentRegionAvail().x / 2.0f;

        // Define fixed number of columns  
        const int fixedColumns = 3;

        // Left Pane: Image Display  
        ImGui::BeginChild("LeftPane", ImVec2(paneWidth, 0), false, ImGuiWindowFlags_AlwaysUseWindowPadding);
        {
            // Zoom Controls  
            // ImGui::SliderFloat("Image Zoom", &imageZoomLevel, 0.5f, 3.0f, "%.1fx");

            // Begin Table for Grid Layout  
            if (ImGui::BeginTable("ImageTable", fixedColumns, ImGuiTableFlags_SizingFixedFit))
            {
                ImGui::TableSetupColumn("Image", ImGuiTableColumnFlags_WidthFixed, 28.0f * imageZoomLevel + 10.0f); // Adjust width as needed  
                // ImGui::TableHeadersRow(); // remove if headers are not needed  

                for (size_t i = 0; i < testSetImages.size(); ++i)
                {
                    if (!ImGui::TableNextColumn())
                        break;

                    const std::vector<float>& pixels = testSetImages[i];


                    // Draw the 28x28 image manually  
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();
                    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
                    ImVec2 image_size = ImVec2(28.0f * imageZoomLevel, 28.0f * imageZoomLevel);
                    ImVec2 image_end = ImVec2(cursor_pos.x + image_size.x, cursor_pos.y + image_size.y);

                    // Draw each pixel as a filled rectangle  
                    for (int row = 0; row < 28; row++)
                    {
                        for (int col = 0; col < 28; col++)
                        {
                            const float val = pixels[row * 28 + col]; // 0..1  
                            const ImColor colour = ImColor(val, val, val, 1.0f);
                            ImVec2 ul = ImVec2(cursor_pos.x + col * imageZoomLevel, cursor_pos.y + row * imageZoomLevel);
                            ImVec2 br = ImVec2(ul.x + imageZoomLevel, ul.y + imageZoomLevel);
                            draw_list->AddRectFilled(ul, br, colour);
                        }
                    }

                    // Add Invisible Button for Interaction  
                    ImGui::SetCursorScreenPos(cursor_pos);
                    ImGui::InvisibleButton(("image_" + std::to_string(i)).c_str(), image_size);

                    // Handle image selection  
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                    {
                        selectedImageIndex = static_cast<int>(i);
                    }

                    // Highlight selected image  
                    if (static_cast<int>(i) == selectedImageIndex)
                    {
                        draw_list->AddRect(cursor_pos, image_end, IM_COL32(255, 255, 0, 255), 0.0f, 0, 2.0f); // Yellow border with 2.0f thickness  
                    }

                    // Tooltip with image index or filename  
                    if (ImGui::IsItemHovered())
                    {
                        ImGui::SetTooltip("Image %zu", i + 1);
                    }
                }
                ImGui::EndTable();
            }
        }
        ImGui::EndChild();

        // Right Pane: Inference Results    
        ImGui::SameLine();

        ImGui::BeginChild("RightPane", ImVec2(paneWidth, 0), false, ImGuiWindowFlags_AlwaysUseWindowPadding);
        {
            // Begin Table for Grid Layout    
            if (ImGui::BeginTable("PredictionTable", fixedColumns, ImGuiTableFlags_SizingFixedFit))
            {
                // Define columns without headers
                ImGui::TableSetupColumn("Prediction", ImGuiTableColumnFlags_WidthFixed, 28.0f * imageZoomLevel + 10.0f); // Adjust width as needed    
                // ImGui::TableHeadersRow(); // Headers removed as per requirement    

                for (size_t i = 0; i < testSetPredictions.size(); ++i)
                {
                    // Move to the next column. If no more columns, move to the next row automatically.
                    if (!ImGui::TableNextColumn())
                        break;

                    int prediction = testSetPredictions[i];

                    // Define box size based on zoom level    
                    ImVec2 box_size = ImVec2(28.0f * imageZoomLevel, 28.0f * imageZoomLevel);

                    ImDrawList* draw_list = ImGui::GetWindowDrawList();
                    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
                    ImVec2 box_end = ImVec2(cursor_pos.x + box_size.x, cursor_pos.y + box_size.y);

                    // Draw black background    
                    draw_list->AddRectFilled(cursor_pos, box_end, IM_COL32(0, 0, 0, 255));

                    // Prepare prediction text    
                    char predText[4];
                    std::snprintf(predText, sizeof(predText), "%d", prediction);

                    // Calculate text size    
                    ImVec2 text_size = ImGui::CalcTextSize(predText);

                    // Draw white text centered within the box    
                    draw_list->AddText(
                        ImVec2(
                            cursor_pos.x + (box_size.x - text_size.x) / 2.0f,
                            cursor_pos.y + (box_size.y - text_size.y) / 2.0f
                        ),
                        IM_COL32(255, 255, 255, 255),
                        predText
                    );

                    // Add Invisible Button for Interaction    
                    ImGui::SetCursorScreenPos(cursor_pos);
                    ImGui::InvisibleButton(("prediction_" + std::to_string(i)).c_str(), box_size);

                    // Handle prediction selection    
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                    {
                        selectedImageIndex = static_cast<int>(i);
                    }

                    // Highlight selected prediction    
                    if (static_cast<int>(i) == selectedImageIndex)
                    {
                        draw_list->AddRect(
                            cursor_pos,
                            box_end,
                            IM_COL32(255, 255, 0, 255),
                            0.0f,
                            0,
                            2.0f
                        ); // Yellow border with 2.0f thickness    
                    }

                    // Tooltip with prediction value    
                    if (ImGui::IsItemHovered())
                    {
                        ImGui::SetTooltip("Confidence: %.3f", (float)testSetConfidences[i]);
                    }
                }
                ImGui::EndTable();
            }
        }
        ImGui::EndChild();

        ImGui::EndChild();
    }
    else
    {
        ImGui::Text("No custom test set loaded.");
    }
    ImGui::End();
}

// Function to display checkpoint details in the UI
void ShowCheckpointDetails(const std::string& filePath)
{
    // Check if the file path is empty
    if (filePath.empty())
    {
        ImGui::Text("No checkpoint selected.");
        return;
    }

    // Attempt to open the file
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        ImGui::Text("Error: Unable to open file: %s", filePath.c_str());
        return;
    }

    try
    {
        // Parse JSON data from the file
        nlohmann::json checkpointData;
        file >> checkpointData;
        file.close();

        // Display general information about the checkpoint
        if (checkpointData.contains("Epoch") && checkpointData["Epoch"].is_number_integer())
        {
            ImGui::Text("Epoch: %d", checkpointData["Epoch"].get<int>());
        }
        else
        {
            ImGui::Text("Epoch: Not found in checkpoint data.");
        }

        if (checkpointData.contains("Accuracy") && checkpointData["Accuracy"].is_number())
        {
            ImGui::Text("Accuracy: %.2f%%", checkpointData["Accuracy"].get<double>() * 100.0);
        }
        else
        {
            ImGui::Text("Accuracy: Not found in checkpoint data.");
        }

        // Optionally display more detailed metadata if available
        if (checkpointData.contains("Metadata") && checkpointData["Metadata"].is_object())
        {
            ImGui::Text("Metadata:");
            for (auto& [key, value] : checkpointData["Metadata"].items())
            {
                ImGui::BulletText("%s: %s", key.c_str(), value.dump().c_str());
            }
        }
    }
    catch (const std::exception& e)
    {
        // Handle JSON parsing errors
        ImGui::Text("Error: Failed to parse checkpoint file: %s", e.what());
    }
}

struct FileMetadata
{
    int epoch;
    double accuracy;
    std::filesystem::file_time_type lastWriteTime;
    bool valid;
};

static std::unordered_map<std::string, FileMetadata> s_checkpointMetadataCache;

static void CreateNetworkWithUIParams(int inputSize, int hiddenLayers, int hiddenSize, int outputSize, int activationElem, int costElem, int finalActElem)
{
    const ActivationType actType = (ActivationType)activationElem;
    const CostType cType = (CostType)costElem;
    const ActivationType finalLayer = (cType == crossEntropy) ? softmax : (ActivationType)finalActElem;

    NeuralNetworkSubsystem& subsystem = NeuralNetworkSubsystem::GetInstance();
    subsystem.InitNeuralNetwork(actType, cType, inputSize, hiddenLayers, hiddenSize, outputSize);
    subsystem.SetVisualizationCallback([](const NeuralNetwork& net)
    {
        showVisualizationPanelWindow = true;
    });
}

// 5.6: Neural Network Controls
void NeuralNetworkControlsWindow(bool* p_open)
{
    if (!ImGui::Begin("Neural Network Controls", p_open))
    {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("##NNControlsTabs"))
    {
        // Tab 1: Architecture
        if (ImGui::BeginTabItem("Architecture"))
        {
            static int inputLayerSize = 784;
            static int numHiddenLayers = 2;
            static int hiddenLayerSize = 128;
            static int outputLayerSize = 10;

            ImGui::InputInt("Input Size", &inputLayerSize);
            ImGui::InputInt("Hidden Layers", &numHiddenLayers);
            ImGui::InputInt("Hidden Layer Size", &hiddenLayerSize);
            ImGui::InputInt("Output Size", &outputLayerSize);

            static int activationElem = (int)ActivationType::ReLU;
            const char* activationNames[] = {"Sigmoid", "ReLU", "LeakyReLU"};
            const char* actName = (activationElem >= 0 && activationElem < 3) ? activationNames[activationElem] : "Unknown";

            // Cost
            static int costElem = (int)CostType::crossEntropy;
            const char* costNames[] = {"Mean Squared Error", "Cross Entropy"};
            const char* costName = (costElem >= 0 && costElem < 2) ? costNames[costElem] : "Unknown";

            ImGui::SliderInt("Activation", &activationElem, 0, 2, actName);
            ImGui::SliderInt("Cost", &costElem, 0, 1, costName);

            // NEW: Final-layer activation control
            static int finalActElem = 0;
            const char* finalActNames[] = {"Sigmoid", "ReLU", "LeakyReLU"};

            if (costElem == (int)crossEntropy)
            {
                // If crossEntropy, force final-layer to softmax
                ImGui::TextDisabled("Final Layer Activation: Softmax (locked for crossEntropy)");
            }
            else
            {
                ImGui::Text("Final Layer Activation:");
                if (ImGui::BeginCombo("##FinalLayerCombo", finalActNames[finalActElem]))
                {
                    for (int i = 0; i < IM_ARRAYSIZE(finalActNames); i++)
                    {
                        bool isSelected = (finalActElem == i);
                        if (ImGui::Selectable(finalActNames[i], isSelected))
                        {
                            finalActElem = i;
                        }
                        if (isSelected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }

            if (ImGui::Button("Create Neural Network (Manual)"))
            {
                CreateNetworkWithUIParams(
                    inputLayerSize, numHiddenLayers,
                    hiddenLayerSize, outputLayerSize,
                    activationElem, costElem, finalActElem
                );

                showDatasetManagementWindow = true;
                showVisualizationPanelWindow = true;
            }

            ImGui::EndTabItem();
        }

        // Tab 2: HyperParameters
        if (ImGui::BeginTabItem("HyperParameters"))
        {
            ImGui::InputFloat("Learning Rate", &HyperParameters::learningRate, 0.001f);
            ImGui::InputInt("Batch Size", &HyperParameters::batchSize);
            ImGui::InputInt("Epochs", &HyperParameters::epochs);
            if (HyperParameters::epochs < 1)
            {
                HyperParameters::epochs = 1;
            }
            ImGui::InputDouble("Weight Decay", &HyperParameters::weightDecay, 0.001, 0.002, "%.5f");

            ImGui::Checkbox("Use Dropout", &HyperParameters::useDropoutRate);
            if (HyperParameters::useDropoutRate)
            {
                ImGui::SliderFloat("Dropout Rate", &HyperParameters::dropoutRate, 0.0f, 1.0f);
            }

            ImGui::Checkbox("Use Gradient Clipping", &HyperParameters::useGradientClipping);
            if (HyperParameters::useGradientClipping)
            {
                float floatGradientThreshold = (float)HyperParameters::gradientClipThreshold;
                ImGui::SliderFloat("Gradient Clipping Threshold", &floatGradientThreshold, 0.0f, 1.0f);
                HyperParameters::gradientClipThreshold = (double)floatGradientThreshold;
            }

            if (ImGui::Button("Reset HyperParameters"))
            {
                HyperParameters::ResetHyperParameters();
            }

            ImGui::EndTabItem();
        }

        // Tab 3: Save/Load
        if (ImGui::BeginTabItem("Save/Load"))
        {
            // -------------------------------
            // 1. Declarations & Refresh Logic
            // -------------------------------
            // Static variables so they persist across frames.
            static bool filesRefreshed = false;
            static int selectedSaveFileIndex = -1;
            static std::string selectedSaveFilePath;
            static std::vector<std::string> saveFiles;

            // Automatically refresh the list whenever this tab is focused again.
            if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) && !filesRefreshed)
            {
                // Clear and repopulate
                saveFiles.clear();
                selectedSaveFileIndex = -1;
                selectedSaveFilePath.clear();

                // Only .json is assumed as your new checkpoint extension
                const std::filesystem::path saveDir = std::filesystem::current_path() / "Saved";

                if (std::filesystem::exists(saveDir))
                {
                    std::vector<std::pair<std::filesystem::file_time_type, std::string>> filesWithTimes;
                    for (const auto& entry : std::filesystem::directory_iterator(saveDir))
                    {
                        if (entry.is_regular_file() && entry.path().extension() == ".json")
                        {
                            // Keep just the filename, e.g. "Network_Epoch10_Acc85_2025-01-20.json"
                            filesWithTimes.push_back({std::filesystem::last_write_time(entry), entry.path().filename().string()});
                        }
                    }

                    std::sort(filesWithTimes.begin(), filesWithTimes.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });

                    saveFiles.clear();
                    for (auto& [timeVal, fileName] : filesWithTimes)
                    {
                        saveFiles.push_back(fileName);
                    }
                }

                filesRefreshed = true;
            }
            // If the user leaves the tab, we want to re-trigger a refresh next time.
            if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows))
            {
                filesRefreshed = false;
            }

            // -------------------------------
            // 2. List of Existing Checkpoints
            // -------------------------------
            ImGui::Text("Existing Checkpoints in 'Saved/' directory:");
            ImGui::Separator();

            // Show the checkpoint files in a list box.
            ImVec2 listBoxSize(-1.0f, ImGui::GetTextLineHeightWithSpacing() * 6.0f); // 12 lines
            if (ImGui::BeginListBox("##CheckpointListBox", listBoxSize))
            {
                for (int i = 0; i < static_cast<int>(saveFiles.size()); i++)
                {
                    bool isSelected = (i == selectedSaveFileIndex);
                    if (ImGui::Selectable(saveFiles[i].c_str(), isSelected))
                    {
                        selectedSaveFileIndex = i;
                        selectedSaveFilePath = (std::filesystem::current_path() / "Saved" / saveFiles[i]).string();
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndListBox();
            }


            // If a file is selected, show the "Load" button
            if (selectedSaveFileIndex >= 0 && selectedSaveFileIndex < (int)saveFiles.size())
            {
                std::string selectedFilePath = (std::filesystem::current_path() / "Saved" / saveFiles[selectedSaveFileIndex]).string();
                // Check last_write_time
                const auto lastWrite = std::filesystem::last_write_time(selectedFilePath);

                auto it = s_checkpointMetadataCache.find(selectedFilePath);
                bool needParse = true;
                if (it != s_checkpointMetadataCache.end())
                {
                    if (it->second.lastWriteTime == lastWrite)
                    {
                        needParse = false;
                    }
                }

                if (needParse)
                {
                    FileMetadata meta{};
                    meta.valid = false;
                    meta.lastWriteTime = lastWrite;
                    try
                    {
                        std::ifstream inFile(selectedFilePath);
                        nlohmann::json j;
                        inFile >> j;
                        inFile.close();

                        meta.epoch = j["TrainingState"]["currentEpoch"].get<int>();
                        meta.accuracy = j["HyperParameters"]["activationType"].get<double>();
                        meta.valid = true;
                    }
                    catch (std::exception e)
                    {
                        LOG(LogLevel::ERROR, e.what());
                    }
                    s_checkpointMetadataCache[selectedFilePath] = meta;
                }

                auto& info = s_checkpointMetadataCache[selectedFilePath];
                if (info.valid)
                {
                    ImGui::Text("Epoch: %d", info.epoch);
                    ImGui::Text("Accuracy: %.2f%%", info.accuracy);
                }
                else
                {
                    ImGui::Text("Error reading metadata or no data cached.");
                }

                if (ImGui::Button("Load Selected Checkpoint"))
                {
                    bool success = NeuralNetworkSubsystem::GetInstance().LoadNetwork(selectedFilePath);
                    if (success)
                    {
                        LOG(LogLevel::INFO, "Successfully loaded checkpoint: " + selectedFilePath);
                    }
                    else
                    {
                        LOG(LogLevel::ERROR, "Failed to load checkpoint: " + selectedFilePath);
                    }
                }
            }

            ImGui::Separator();

            // -------------------------------
            // 3. Export (Save) the Network
            // -------------------------------
            ImGui::Text("Export Current Network to a new JSON file:");
            if (ImGui::Button("Export Network"))
            {
                // Create an auto-named filename, e.g. "Network_Epoch5_Acc85_2025-01-25_20-40-55.json"
                int currentEpoch = NeuralNetworkSubsystem::GetInstance().currentEpochAtomic.load();
                std::string timestamp = NeuralNetworkUtility::GetTimeStampWithAnnotations();

                double testAccuracy = NeuralNetworkSubsystem::GetInstance().EvaluateTestSet(); // need to make this use std::future or wait some other way.
                if (testAccuracy == 0.0)
                {
                    testAccuracy = NeuralNetworkSubsystem::GetInstance().currentAccuracyAtomic.load() * 100.0;
                }
                std::string autoFilename =
                    "Network_Epoch" + std::to_string(currentEpoch) +
                    "_Acc" + std::to_string((int)testAccuracy * 100.0) + "_" +
                    timestamp + ".json";

                std::string savePath =
                    (std::filesystem::current_path() / "Saved" / autoFilename).string();

                bool savedOk = NeuralNetworkSubsystem::GetInstance().SaveNetwork(savePath);
                if (savedOk)
                {
                    LOG(LogLevel::INFO, "Exported to: " + autoFilename);
                    // Force a refresh next frame so the new file appears in the list
                    filesRefreshed = false;
                }
                else
                {
                    LOG(LogLevel::ERROR, "Failed to export network: " + autoFilename);
                }
            }

            ImGui::EndTabItem();
        }
    }
    ImGui::EndTabBar();
    ImGui::End();
}

// 5.7: Default Windows aggregator
void DefaultWindows()
{
    if (showDatasetManagementWindow)
    {
        DatasetManagementWindow(&showDatasetManagementWindow,
                                NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork());
    }
    if (showNeuralNetworkControlsWindow)
    {
        NeuralNetworkControlsWindow(&showNeuralNetworkControlsWindow);
    }
    if (showVisualizationPanelWindow)
    {
        VisualizationPanelWindow(&showVisualizationPanelWindow,
                                 NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork());
    }
    if (showAdvancedEditingWindow)
    {
        AdvancedEditingWindow(&showAdvancedEditingWindow,
                              NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork());
    }
}

//--------------------------------------------------------------------
// [SECTION 6] Vulkan Setup / Teardown
//--------------------------------------------------------------------
#ifdef IMGUI_VULKAN_DEBUG_REPORT
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(
    VkDebugReportFlagsEXT /*flags*/,
    VkDebugReportObjectTypeEXT objectType,
    uint64_t object, size_t location,
    int32_t messageCode, const char* pLayerPrefix,
    const char* pMessage, void* /*pUserData*/)
{
    fprintf(stderr, "[vulkan] Debug report from ObjectType: %i\nMessage: %s\n\n",
            objectType, pMessage);
    return VK_FALSE;
}
#endif

static bool IsExtensionAvailable(const ImVector<VkExtensionProperties>& props, const char* extension)
{
    for (auto& p : props)
        if (strcmp(p.extensionName, extension) == 0)
            return true;
    return false;
}

static VkPhysicalDevice SetupVulkan_SelectPhysicalDevice()
{
    uint32_t gpu_count;
    VkResult err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, nullptr);
    check_vk_result(err);
    IM_ASSERT(gpu_count > 0);

    ImVector<VkPhysicalDevice> gpus;
    gpus.resize(gpu_count);
    err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, gpus.Data);
    check_vk_result(err);

    for (VkPhysicalDevice& device : gpus)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            return device;
    }
    return gpus[0];
}

static void SetupVulkan(ImVector<const char*> instance_extensions)
{
    VkResult err;

    // [1] Create Vulkan Instance
    {
        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

        uint32_t propCount;
        ImVector<VkExtensionProperties> props;
        vkEnumerateInstanceExtensionProperties(nullptr, &propCount, nullptr);
        props.resize(propCount);
        err = vkEnumerateInstanceExtensionProperties(nullptr, &propCount, props.Data);
        check_vk_result(err);

        if (IsExtensionAvailable(props, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if (IsExtensionAvailable(props, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
        {
            instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

#ifdef IMGUI_VULKAN_DEBUG_REPORT
        const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
        create_info.enabledLayerCount = 1;
        create_info.ppEnabledLayerNames = layers;
        instance_extensions.push_back("VK_EXT_debug_report");
#endif

        create_info.enabledExtensionCount = (uint32_t)instance_extensions.Size;
        create_info.ppEnabledExtensionNames = instance_extensions.Data;
        err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
        check_vk_result(err);

#ifdef IMGUI_VULKAN_DEBUG_REPORT
        auto vkCreateDebugReportCallbackEXT =
            (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
                g_Instance, "vkCreateDebugReportCallbackEXT");
        IM_ASSERT(vkCreateDebugReportCallbackEXT != nullptr);
        VkDebugReportCallbackCreateInfoEXT dbg_info = {};
        dbg_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        dbg_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
            VK_DEBUG_REPORT_WARNING_BIT_EXT |
            VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        dbg_info.pfnCallback = debug_report;
        dbg_info.pUserData = nullptr;
        err = vkCreateDebugReportCallbackEXT(g_Instance, &dbg_info, g_Allocator, &g_DebugReport);
        check_vk_result(err);
#endif
    }

    // [2] Select Physical Device
    g_PhysicalDevice = SetupVulkan_SelectPhysicalDevice();

    // [3] Select Graphics Queue
    {
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, nullptr);
        VkQueueFamilyProperties* queues =
            (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * count);
        vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, queues);
        for (uint32_t i = 0; i < count; i++)
        {
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                g_QueueFamily = i;
                break;
            }
        }
        free(queues);
        IM_ASSERT(g_QueueFamily != (uint32_t)-1);
    }

    // [4] Create Logical Device
    {
        ImVector<const char*> devExt;
        devExt.push_back("VK_KHR_swapchain");

        uint32_t devPropCount;
        ImVector<VkExtensionProperties> devProps;
        vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &devPropCount, nullptr);
        devProps.resize(devPropCount);
        vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &devPropCount, devProps.Data);

#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        if (IsExtensionAvailable(devProps, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME))
            devExt.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

        const float queue_priority[] = {1.0f};
        VkDeviceQueueCreateInfo queue_info[1] = {};
        queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[0].queueFamilyIndex = g_QueueFamily;
        queue_info[0].queueCount = 1;
        queue_info[0].pQueuePriorities = queue_priority;

        VkDeviceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = 1;
        create_info.pQueueCreateInfos = queue_info;
        create_info.enabledExtensionCount = (uint32_t)devExt.Size;
        create_info.ppEnabledExtensionNames = devExt.Data;

        err = vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
        check_vk_result(err);
        vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
    }

    // [5] Descriptor Pool
    {
        VkDescriptorPoolSize pool_sizes[] =
        {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        err = vkCreateDescriptorPool(g_Device, &pool_info, g_Allocator, &g_DescriptorPool);
        check_vk_result(err);
    }
}

static void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height)
{
    wd->Surface = surface;

    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_QueueFamily, wd->Surface, &res);
    if (res != VK_TRUE)
    {
        fprintf(stderr, "Error no WSI support on physical device\n");
        exit(-1);
    }

    // Surface format
    const VkFormat requestSurfaceImageFormat[] = {
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8_UNORM,
        VK_FORMAT_R8G8B8_UNORM
    };
    const VkColorSpaceKHR reqColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;

    wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        g_PhysicalDevice, wd->Surface,
        requestSurfaceImageFormat,
        (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat),
        reqColorSpace
    );

#ifdef IMGUI_UNLIMITED_FRAME_RATE
    VkPresentModeKHR present_modes[] = {
        VK_PRESENT_MODE_MAILBOX_KHR,
        VK_PRESENT_MODE_IMMEDIATE_KHR,
        VK_PRESENT_MODE_FIFO_KHR
    };
#else
    VkPresentModeKHR present_modes[] = {
        VK_PRESENT_MODE_FIFO_KHR
    };
#endif

    wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        g_PhysicalDevice, wd->Surface,
        &present_modes[0],
        IM_ARRAYSIZE(present_modes)
    );

    IM_ASSERT(g_MinImageCount >= 2);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        g_Instance, g_PhysicalDevice, g_Device,
        wd, g_QueueFamily, g_Allocator,
        width, height, g_MinImageCount
    );
}

static void CleanupVulkanWindow()
{
    ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData, g_Allocator);
}

static void CleanupVulkan()
{
    vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

#ifdef IMGUI_VULKAN_DEBUG_REPORT
    auto vkDestroyDebugReportCallbackEXT =
        (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
            g_Instance, "vkDestroyDebugReportCallbackEXT");
    if (vkDestroyDebugReportCallbackEXT)
        vkDestroyDebugReportCallbackEXT(g_Instance, g_DebugReport, g_Allocator);
#endif

    vkDestroyDevice(g_Device, g_Allocator);
    vkDestroyInstance(g_Instance, g_Allocator);
}

static void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
{
    VkResult err;
    VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;

    err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX,
                                image_acquired_semaphore, VK_NULL_HANDLE,
                                &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    {
        g_SwapChainRebuild = true;
        return;
    }
    check_vk_result(err);

    ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
    {
        err = vkWaitForFences(g_Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);
        check_vk_result(err);
        err = vkResetFences(g_Device, 1, &fd->Fence);
        check_vk_result(err);
    }
    {
        err = vkResetCommandPool(g_Device, fd->CommandPool, 0);
        check_vk_result(err);

        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
        check_vk_result(err);
    }
    {
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = wd->RenderPass;
        info.framebuffer = fd->Framebuffer;
        info.renderArea.extent.width = wd->Width;
        info.renderArea.extent.height = wd->Height;
        info.clearValueCount = 1;
        info.pClearValues = &wd->ClearValue;
        vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    // Record dear imgui primitives
    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

    vkCmdEndRenderPass(fd->CommandBuffer);

    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &image_acquired_semaphore;
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &fd->CommandBuffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_complete_semaphore;

        err = vkEndCommandBuffer(fd->CommandBuffer);
        check_vk_result(err);
        err = vkQueueSubmit(g_Queue, 1, &submit_info, fd->Fence);
        check_vk_result(err);
    }
}

static void FramePresent(ImGui_ImplVulkanH_Window* wd)
{
    if (g_SwapChainRebuild)
        return;
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &wd->Swapchain;
    info.pImageIndices = &wd->FrameIndex;

    VkResult err = vkQueuePresentKHR(g_Queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    {
        g_SwapChainRebuild = true;
        return;
    }
    check_vk_result(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount;
}

//--------------------------------------------------------------------
// [SECTION 7] Main
//--------------------------------------------------------------------
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void check_vk_result(VkResult err)
{
    if (err == 0) return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0) abort();
}

int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

    // Create window with Vulkan context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1880, 920, "MNIST Full Flow Example", nullptr, nullptr);
    if (!glfwVulkanSupported())
    {
        printf("GLFW: Vulkan Not Supported\n");
        return 1;
    }

    // Collect required extensions
    ImVector<const char*> extensions;
    uint32_t extensions_count = 0;
    const char** glfw_ext = glfwGetRequiredInstanceExtensions(&extensions_count);
    for (uint32_t i = 0; i < extensions_count; i++)
        extensions.push_back(glfw_ext[i]);

    // Initialize Vulkan
    SetupVulkan(extensions);

    // Create Window Surface
    VkSurfaceKHR surface;
    VkResult err = glfwCreateWindowSurface(g_Instance, window, g_Allocator, &surface);
    check_vk_result(err);

    // Create Framebuffers
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
    SetupVulkanWindow(wd, surface, w, h);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::StyleColorsDark();

    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        ImGui::GetStyle().WindowRounding = 0.0f;
        ImGui::GetStyle().Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = g_Instance;
    init_info.PhysicalDevice = g_PhysicalDevice;
    init_info.Device = g_Device;
    init_info.QueueFamily = g_QueueFamily;
    init_info.Queue = g_Queue;
    init_info.PipelineCache = g_PipelineCache;
    init_info.DescriptorPool = g_DescriptorPool;
    init_info.Subpass = 0;
    init_info.MinImageCount = g_MinImageCount;
    init_info.ImageCount = wd->ImageCount;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = g_Allocator;
    init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);

    // Optionally load fonts
    // ImGui::GetIO().Fonts->AddFontDefault();

    // Upload fonts
    {
        VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
        VkCommandBuffer cmd_buf;

        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = command_pool;
        alloc_info.commandBufferCount = 1;

        err = vkAllocateCommandBuffers(g_Device, &alloc_info, &cmd_buf);
        check_vk_result(err);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(cmd_buf, &begin_info);
        check_vk_result(err);
        // If needed: ImGui_ImplVulkan_CreateFontsTexture(cmd_buf);

        VkSubmitInfo end_info = {};
        end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        end_info.commandBufferCount = 1;
        end_info.pCommandBuffers = &cmd_buf;
        err = vkEndCommandBuffer(cmd_buf);
        check_vk_result(err);
        err = vkQueueSubmit(g_Queue, 1, &end_info, VK_NULL_HANDLE);
        check_vk_result(err);
        err = vkDeviceWaitIdle(g_Device);
        check_vk_result(err);
    }

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Rebuild swapchain?
        if (g_SwapChainRebuild)
        {
            glfwGetFramebufferSize(window, &w, &h);
            if (w > 0 && h > 0)
            {
                ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
                ImGui_ImplVulkanH_CreateOrResizeWindow(
                    g_Instance, g_PhysicalDevice, g_Device,
                    &g_MainWindowData,
                    g_QueueFamily, g_Allocator,
                    w, h, g_MinImageCount
                );
                g_MainWindowData.FrameIndex = 0;
                g_SwapChainRebuild = false;
            }
        }

        // Start the Dear ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Show the default windows
        DefaultWindows();

        // Render
        ImGui::Render();
        ImDrawData* main_draw_data = ImGui::GetDrawData();
        bool minimized = (main_draw_data->DisplaySize.x <= 0.0f ||
            main_draw_data->DisplaySize.y <= 0.0f);

        wd->ClearValue.color.float32[0] = 0.0f;
        wd->ClearValue.color.float32[1] = 0.0f;
        wd->ClearValue.color.float32[2] = 0.0f;
        wd->ClearValue.color.float32[3] = 1.0f;

        if (!minimized)
            FrameRender(wd, main_draw_data);

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }
        if (!minimized)
            FramePresent(wd);
    }

    // Cleanup
    err = vkDeviceWaitIdle(g_Device);
    check_vk_result(err);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    CleanupVulkanWindow();
    CleanupVulkan();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
