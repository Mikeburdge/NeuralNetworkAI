#include "cstdio"          // printf, fprintf
#include "cstdlib"         // abort
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <filesystem>
#include <string>
#include <stb_image.h>
#include <GLFW/glfw3.h>
#include <ImGuiFileDialog/ImGuiFileDialog.h>
#include <vulkan/vulkan.h>

#include "core/HyperParameters.h"
#include "core/NeuralNetwork.h"
#include "core/VisualisationUtility.h"

#include "dataloader/MNISTDataSet.h"
#include "subsystems/NeuralNetworkSubsystem.h"

// [Win32] Our example includes a copy of glfw3.lib pre-compiled ...
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

float minCircleSizeValue = 5.0f;
float maxCircleSizeValue = 50.0f;
float circleThicknessValue = 1.0f;
float minLineThicknessValue = 1.0f;
bool drawLineConnections = true;
bool drawWeights = false;
bool activeNeuronCanPulse = true;
int hoveredWeightIndex = -1;
int clickedWeightIndex = -1;

std::string filePathName;
std::string filePath;

//--------------------------------------------------------------------
// Forward refs
//--------------------------------------------------------------------
static void glfw_error_callback(int error, const char* description);
static void check_vk_result(VkResult err);

//--------------------------------------------------------------------
// [SECTION 3] A simple texture struct, if you need it
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

// 5.1 Advanced Editing
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

// 5.2 Visualization Panel
static float CalculateMaxCircleSize(const ImVec2& winSize, int numLayers, float maxCircleSizeValue)
{
    return std::min(std::min(winSize.x, winSize.y) / (numLayers * 2.0f), maxCircleSizeValue);
}

void VisualizationPanelWindow(bool* p_open, const NeuralNetwork& network)
{
    if (!ImGui::Begin("Neural Network Visualization (Panel)", p_open))
    {
        ImGui::End();
        return;
    }

    const ImVec2 windowSize = ImGui::GetWindowSize();
    const ImVec2 windowPos = ImGui::GetWindowPos();
    const float windowScrollY = ImGui::GetScrollY();
    const float maxCircleSize = CalculateMaxCircleSize(windowSize,
                                                       (int)network.layers.size(),
                                                       maxCircleSizeValue);
    const float layerSpacing = windowSize.x / (network.layers.size() + 1);

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

    // Render Layers
    for (int layerIndex = 0; layerIndex < (int)network.layers.size(); ++layerIndex)
    {
        const Layer& layer = network.layers[layerIndex];
        float layerPosX = (layerIndex + 1) * layerSpacing;
        int numNeurons = layer.numNeurons;
        float neuronSpacing = windowSize.y / (numNeurons + 1);

        for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex)
        {
            float circleSize = std::min(maxCircleSize, neuronSpacing * 0.5f);
            float posX = layerPosX;
            float posY = (neuronIndex + 1) * neuronSpacing;

            ImGui::SetCursorPos(ImVec2(posX - circleSize, posY - circleSize));
            ImGui::BeginGroup();
            ImGui::PushID((void*)((uintptr_t(layerIndex) << 16) | uintptr_t(neuronIndex)));

            const Neuron currentNeuron = layer.neurons[neuronIndex];
            ImColor colour = VisualisationUtility::GetActivationColour(
                (float)currentNeuron.ActivationValue, HyperParameters::activationType);

            ImVec2 circlePos(
                ImGui::GetCursorScreenPos().x + circleSize,
                ImGui::GetCursorScreenPos().y + circleSize
            );

            float pulseSize = circleSize;
            if (activeNeuronCanPulse && currentNeuron.ActivationValue > 0.5f)
            {
                pulseSize += (sinf(ImGui::GetTime() * 5.0f) * 2.0f);
            }

            ImGui::GetWindowDrawList()->AddCircle(
                circlePos, pulseSize, colour, 12, 1.0f
            );

            // Hover detection
            const ImVec2 mousePos = ImGui::GetMousePos();
            bool mouseHoveringNeuron =
                VisualisationUtility::PointToCircleCollisionCheck(mousePos, circlePos, circleSize);

            // Draw activation text
            char buf[32];
            if (mouseHoveringNeuron) std::snprintf(buf, sizeof(buf), "%.3f", currentNeuron.ActivationValue);
            else std::snprintf(buf, sizeof(buf), "%.1f", currentNeuron.ActivationValue);

            ImVec2 textSize = ImGui::CalcTextSize(buf);
            ImVec2 textPos = ImVec2(posX - textSize.x * 0.5f, posY - textSize.y * 0.5f);
            ImGui::SetCursorPos(textPos);
            ImGui::Text("%s", buf);

            ImGui::PopID();
            ImGui::EndGroup();

            // Draw connections from previous layer
            if (layerIndex > 0 && drawLineConnections)
            {
                float prevLayerPosX = layerPosX - layerSpacing;
                const Layer& prevLayer = network.layers[layerIndex - 1];
                int numNeuronsPrev = prevLayer.numNeurons;
                float neuronSpacingPrev = windowSize.y / (numNeuronsPrev + 1);
                float prevCircleSize = std::min(maxCircleSize, neuronSpacingPrev * 0.5f);

                if ((int)layer.weights.size() == numNeurons &&
                    (int)layer.weights[neuronIndex].size() == numNeuronsPrev)
                {
                    for (int prevNeuronIndex = 0; prevNeuronIndex < numNeuronsPrev; ++prevNeuronIndex)
                    {
                        float weight = (float)layer.weights[neuronIndex][prevNeuronIndex];
                        float thickness = std::max(minLineThicknessValue,
                                                   1.0f + std::min(4.0f, fabs(weight) / 5.0f));

                        ImVec2 lineStart(
                            prevLayerPosX + prevCircleSize + windowPos.x,
                            (prevNeuronIndex + 1) * neuronSpacingPrev + windowPos.y - windowScrollY
                        );
                        ImVec2 lineEnd(
                            posX - circleSize + windowPos.x,
                            posY + windowPos.y - windowScrollY
                        );

                        bool isHovered =
                        (VisualisationUtility::DistanceToLineSegment(mousePos, lineStart, lineEnd)
                            < (minLineThicknessValue + 2.0f));
                        bool isClicked = isHovered && ImGui::IsMouseClicked(0);

                        if (isHovered) hoveredWeightIndex = neuronIndex + numNeuronsPrev + prevNeuronIndex;
                        if (isClicked) clickedWeightIndex = hoveredWeightIndex;

                        const ImColor lineColor = ImColor(isHovered ? ImVec4(255, 255, 0, 255) : VisualisationUtility::GetWeightColor(weight));

                        ImGui::GetWindowDrawList()->AddLine(lineStart, lineEnd, lineColor, thickness);

                        if (drawWeights || isHovered || mouseHoveringNeuron)
                        {
                            lineInfos.emplace_back(lineStart, lineEnd, weight, isHovered);
                        }
                    }
                }
            }
        }
    }

    // Weight text near lines
    for (auto& info : lineInfos)
    {
        VisualisationUtility::DrawWeightText(info.lineStart, info.lineEnd, info.weight);
    }

    ImGui::End();

    // Another window for customization
    ImGui::Begin("Visualization Customization");
    ImGui::SliderFloat("Min Circle Size", &minCircleSizeValue, 1.0f, 10.0f);
    ImGui::SliderFloat("Max Circle Size", &maxCircleSizeValue, 10.0f, 100.0f);
    ImGui::SliderFloat("Circle Thickness", &circleThicknessValue, 1.0f, 5.0f);
    ImGui::Checkbox("Draw Lines", &drawLineConnections);
    ImGui::SliderFloat("Min Line Thickness", &minLineThicknessValue, 1.0f, 5.0f);
    ImGui::Checkbox("Draw Weights", &drawWeights);
    ImGui::Checkbox("Pulsating Neurons", &activeNeuronCanPulse);
    ImGui::End();
}

// 5.3 Dataset Management
MyTextureData loadedInputImage;

void DatasetManagementWindow(bool* p_open, NeuralNetwork& network)
{
    if (!ImGui::Begin("Dataset Management", p_open))
    {
        ImGui::End();
        return;
    }

    ImGui::TextWrapped("This panel lets you load MNIST data and train automatically.");

    // If the data is loaded, show how many samples
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

    // Provide a button that triggers the "full flow"
    if (ImGui::Button("Train MNIST (Full Process)"))
    {
        NeuralNetworkSubsystem::GetInstance().TrainOnMNISTFullProcess();
    }

    ImGui::Separator();
    ImGui::TextWrapped("If you want to do things manually (browsing, etc.), you can still do so here...");

    // Typically you'd have manual file loading UI, but skipping for brevity...
    // ...

    ImGui::End();
}

// 5.4 Neural Network Controls Window
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
            static int numHiddenLayers = 1;
            static int hiddenLayerSize = 128;
            static int outputLayerSize = 10;

            ImGui::InputInt("Input Size", &inputLayerSize);
            ImGui::InputInt("Hidden Layers", &numHiddenLayers);
            ImGui::InputInt("Hidden Layer Size", &hiddenLayerSize);
            ImGui::InputInt("Output Size", &outputLayerSize);

            // Activation
            static int activationElem = (int)sigmoid;
            const char* activationNames[] = {"Sigmoid", "Sigmoid Derivative", "ReLU"};
            const char* actName = (activationElem >= 0 && activationElem < Activation_Count) ? activationNames[activationElem] : "Unknown";
            ImGui::SliderInt("Activation", &activationElem, 0, Activation_Count - 1, actName);

            // Cost
            static int costElem = (int)crossEntropy;
            const char* costNames[] = {"Mean Squared Error", "Cross Entropy"};
            const char* costName = (costElem >= 0 && costElem < cost_Count) ? costNames[costElem] : "Unknown";
            ImGui::SliderInt("Cost", &costElem, 0, cost_Count - 1, costName);

            if (ImGui::Button("Create Neural Network (Manual)"))
            {
                ActivationType actType = (ActivationType)activationElem;
                CostType cType = (CostType)costElem;

                auto& subsystem = NeuralNetworkSubsystem::GetInstance();
                subsystem.InitNeuralNetwork(actType, cType,
                                            inputLayerSize,
                                            numHiddenLayers,
                                            hiddenLayerSize,
                                            outputLayerSize);
                subsystem.SetVisualizationCallback([](const NeuralNetwork& net)
                {
                    showVisualizationPanelWindow = true;
                });

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
            ImGui::InputDouble("Momentum", &HyperParameters::momentum, 0.1, 0.2, "%.2f");
            ImGui::InputDouble("Weight Decay", &HyperParameters::weightDecay, 0.001, 0.002, "%.3f");

            ImGui::Checkbox("Use Dropout", &HyperParameters::useDropoutRate);
            if (HyperParameters::useDropoutRate)
            {
                ImGui::SliderFloat("Dropout Rate", &HyperParameters::dropoutRate, 0.0f, 1.0f);
            }

            if (ImGui::Button("Reset HyperParameters"))
            {
                HyperParameters::ResetHyperParameters();
            }

            ImGui::EndTabItem();
        }
    }
    ImGui::EndTabBar();
    ImGui::End();
}

// 5.5 DefaultWindows aggregator
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
    VkSemaphore image_acquired_sem = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_sem = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;

    err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX,
                                image_acquired_sem,
                                VK_NULL_HANDLE,
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

    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

    vkCmdEndRenderPass(fd->CommandBuffer);

    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &image_acquired_sem;
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &fd->CommandBuffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_complete_sem;

        VkResult errEnd = vkEndCommandBuffer(fd->CommandBuffer);
        check_vk_result(errEnd);

        errEnd = vkQueueSubmit(g_Queue, 1, &submit_info, fd->Fence);
        check_vk_result(errEnd);
    }
}

static void FramePresent(ImGui_ImplVulkanH_Window* wd)
{
    if (g_SwapChainRebuild) return;

    VkSemaphore render_complete_sem = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_sem;
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

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "MNIST Full Flow Example", nullptr, nullptr);
    if (!glfwVulkanSupported())
    {
        printf("GLFW: Vulkan Not Supported\n");
        return 1;
    }

    ImVector<const char*> extensions;
    uint32_t extensions_count = 0;
    const char** glfw_ext = glfwGetRequiredInstanceExtensions(&extensions_count);
    for (uint32_t i = 0; i < extensions_count; i++)
        extensions.push_back(glfw_ext[i]);

    // Setup Vulkan
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

    // Setup ImGui
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

        // Here you could do ImGui_ImplVulkan_CreateFontsTexture(cmd_buf) if needed

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

    // Main Loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (g_SwapChainRebuild)
        {
            glfwGetFramebufferSize(window, &w, &h);
            if (w > 0 && h > 0)
            {
                ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
                ImGui_ImplVulkanH_CreateOrResizeWindow(
                    g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData,
                    g_QueueFamily, g_Allocator, w, h, g_MinImageCount
                );
                g_MainWindowData.FrameIndex = 0;
                g_SwapChainRebuild = false;
            }
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        DefaultWindows();

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
