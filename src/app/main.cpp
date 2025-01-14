#include "cstdio"          // printf, fprintf
#include "cstdlib"         // abort
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "string"
#include "stb_image.h"
#include "GLFW/glfw3.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"
#include "vulkan/vulkan.h"
#include "core/HyperParameters.h"

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010...
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

//#define IMGUI_UNLIMITED_FRAME_RATE
#ifdef _DEBUG
#define IMGUI_VULKAN_DEBUG_REPORT
#endif

//--------------------------------------------------------------------
// [SECTION 1] Global Vulkan Data/Variables
//--------------------------------------------------------------------
static VkAllocationCallbacks* g_Allocator           = nullptr;
static VkInstance             g_Instance            = VK_NULL_HANDLE;
static VkPhysicalDevice       g_PhysicalDevice      = VK_NULL_HANDLE;
static VkDevice               g_Device              = VK_NULL_HANDLE;
static uint32_t               g_QueueFamily         = (uint32_t)-1;
static VkQueue                g_Queue               = VK_NULL_HANDLE;
static VkDebugReportCallbackEXT g_DebugReport       = VK_NULL_HANDLE;
static VkPipelineCache        g_PipelineCache       = VK_NULL_HANDLE;
static VkDescriptorPool       g_DescriptorPool      = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static int                      g_MinImageCount      = 2;
static bool                     g_SwapChainRebuild   = false;

//--------------------------------------------------------------------
// [SECTION 2] Global UI State Toggles (Windows on/off) & Variables
//--------------------------------------------------------------------
// Windows in the new design
bool showDatasetManagementWindow      = false;  // was showNeuralNetworkStartWindow
bool showNeuralNetworkControlsWindow  = true;   // merges old “Customisation” + “HyperParameter”
bool showVisualizationPanelWindow     = false;  // was showNeuralNetworkWindow
bool showAdvancedEditingWindow        = true;   // was showWeightsAndBiasesWindow

// Visualization settings
float minCircleSizeValue      = 5.0f;
float maxCircleSizeValue      = 50.0f;
float circleThicknessValue    = 1.0f;
float minLineThicknessValue   = 1.0f;
bool  drawLineConnections     = true;
bool  drawWeights            = false;
bool  activeNeuronCanPulse   = true;
int   hoveredWeightIndex     = -1;
int   clickedWeightIndex     = -1;

// File loading
std::string   filePathName;
std::string   filePath;

// Forward references to avoid clutter in main()
static void   glfw_error_callback(int error, const char* description);
static void   check_vk_result(VkResult err);

//--------------------------------------------------------------------
// [SECTION 3] Headers for Neural Network / Vulkan-related Classes
//--------------------------------------------------------------------
#include "core/NeuralNetwork.h"
#include "core/VisualisationUtility.h"
#include "subsystems/NeuralNetworkSubsystem.h"

// A struct to manage data related to one image in Vulkan
struct MyTextureData
{
    VkDescriptorSet DS;
    int Width;
    int Height;
    int Channels;
    VkImageView     ImageView;
    VkImage         Image;
    VkDeviceMemory  ImageMemory;
    VkSampler       Sampler;
    VkBuffer        UploadBuffer;
    VkDeviceMemory  UploadBufferMemory;

    MyTextureData() { memset(this, 0, sizeof(*this)); }
};

//--------------------------------------------------------------------
// [SECTION 4] Vulkan Helper Functions / Texture Loading
//--------------------------------------------------------------------
uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(g_PhysicalDevice, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    return 0xFFFFFFFF; // Unable to find memoryType
}

void RemoveTexture(MyTextureData* tex_data)
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
// [SECTION 5] Windows/ImGui UI Code
//--------------------------------------------------------------------

// 5.1  Advanced Editing Window (was WeightsAndBiasesWindow)
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
            float biasVal = static_cast<float>(network.layers[layerIndex].biases[neuronIndex]);
            ImGui::SliderFloat(
                ("Bias##" + std::to_string(layerIndex) + "_" + std::to_string(neuronIndex)).c_str(),
                &biasVal, -2.0f, 2.0f, "%.1f"
            );
            network.layers[layerIndex].biases[neuronIndex] = static_cast<double>(biasVal);
        }

        // Weights
        ImGui::Text("Weights:");
        if (!layer.weights.empty())
        {
            for (size_t neuronIndex = 0; neuronIndex < layer.weights.size(); ++neuronIndex)
            {
                for (size_t weightIndex = 0; weightIndex < layer.weights[neuronIndex].size(); ++weightIndex)
                {
                    float weightVal = static_cast<float>(network.layers[layerIndex].weights[neuronIndex][weightIndex]);
                    ImGui::SliderFloat(
                        ("Weight##" + std::to_string(layerIndex) + "_" + 
                         std::to_string(neuronIndex) + "_" +
                         std::to_string(weightIndex)).c_str(),
                        &weightVal, -5.0f, 5.0f, "%.2f"
                    );
                    network.layers[layerIndex].weights[neuronIndex][weightIndex] = static_cast<double>(weightVal);
                }
            }
        }

        ImGui::Separator();
    }

    ImGui::End();
}

// 5.2 Visualization Panel Window (was NeuralNetworkWindow)
static float CalculateMaxCircleSize(const ImVec2& windowSize, int numLayers, float maxCircleSizeValue)
{
    return std::min(std::min(windowSize.x, windowSize.y) / (numLayers * 2.0f), maxCircleSizeValue);
}

void VisualizationPanelWindow(bool* p_open, const NeuralNetwork& network)
{
    if (!ImGui::Begin("Neural Network Visualization (Panel)", p_open))
    {
        ImGui::End();
        return;
    }

    const ImVec2 windowSize   = ImGui::GetWindowSize();
    const ImVec2 windowPos    = ImGui::GetWindowPos();
    const float  windowScrollY= ImGui::GetScrollY();

    const float maxCircleSize = CalculateMaxCircleSize(windowSize, 
                                                       static_cast<int>(network.layers.size()),
                                                       maxCircleSizeValue);
    const float layerSpacing  = windowSize.x / (network.layers.size() + 1);

    struct LineInfo
    {
        LineInfo(const ImVec2& s, const ImVec2& e, float w, bool hov)
            : lineStart(s), lineEnd(e), weight(w), isHovered(hov) {}
        ImVec2 lineStart;
        ImVec2 lineEnd;
        float  weight;
        bool   isHovered;
    };

    std::vector<LineInfo> lineInfos;

    // Draw Layers
    for (int layerIndex = 0; layerIndex < (int)network.layers.size(); ++layerIndex)
    {
        const Layer& layer    = network.layers[layerIndex];
        const float layerPosX = (layerIndex + 1) * layerSpacing;
        const int   numNeurons= layer.numNeurons;
        const float neuronSpacing = windowSize.y / (numNeurons + 1);

        for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex)
        {
            const float circleSize = std::min(maxCircleSize, neuronSpacing * 0.5f);
            const float posX       = layerPosX;
            const float posY       = (neuronIndex + 1) * neuronSpacing;

            ImGui::SetCursorPos(ImVec2(posX - circleSize, posY - circleSize));
            ImGui::BeginGroup();
            ImGui::PushID((void*)((uintptr_t(layerIndex) << 16) | uintptr_t(neuronIndex)));

            const Neuron currentNeuron = layer.neurons[neuronIndex];
            ImColor colour = VisualisationUtility::GetActivationColour(
                currentNeuron.ActivationValue,
                HyperParameters::activationType // or the actual type used
            );

            const ImVec2 circlePos(
                ImGui::GetCursorScreenPos().x + circleSize,
                ImGui::GetCursorScreenPos().y + circleSize
            );

            // Pulsating
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

            // Display activation value in center
            char buffer[32];
            if (mouseHoveringNeuron)
                std::snprintf(buffer, sizeof(buffer), "%.3f", currentNeuron.ActivationValue);
            else
                std::snprintf(buffer, sizeof(buffer), "%.1f", currentNeuron.ActivationValue);
            
            const ImVec2 textSize = ImGui::CalcTextSize(buffer);
            const ImVec2 textPos  = ImVec2(posX - textSize.x * 0.5f, posY - textSize.y * 0.5f);
            ImGui::SetCursorPos(textPos);
            ImGui::Text("%s", buffer);

            ImGui::PopID();
            ImGui::EndGroup();

            // Draw connections from previous layer
            if (layerIndex > 0 && drawLineConnections)
            {
                const float prevLayerPosX  = layerPosX - layerSpacing;
                const Layer& previousLayer = network.layers[layerIndex - 1];
                const int    numNeuronsPrev= previousLayer.numNeurons;
                const float  neuronSpacingPrevLayer = windowSize.y / (numNeuronsPrev + 1);
                const float  prevCircleSize = std::min(maxCircleSize, neuronSpacingPrevLayer * 0.5f);

                // ensure weights are valid
                if ((int)layer.weights.size() == numNeurons &&
                    (int)layer.weights[neuronIndex].size() == numNeuronsPrev)
                {
                    for (int prevNeuronIndex = 0; prevNeuronIndex < numNeuronsPrev; ++prevNeuronIndex)
                    {
                        const float prevPosY = (prevNeuronIndex + 1) * neuronSpacingPrevLayer;
                        float weight         = layer.weights[neuronIndex][prevNeuronIndex];
                        float thickness      = 
                            std::max(minLineThicknessValue, 1.0f + std::min(4.0f, fabs(weight) / 5.0f));

                        ImVec2 lineStart = ImVec2(
                            prevLayerPosX + prevCircleSize + windowPos.x,
                            prevPosY + windowPos.y - windowScrollY
                        );
                        ImVec2 lineEnd   = ImVec2(
                            posX - circleSize + windowPos.x,
                            posY + windowPos.y - windowScrollY
                        );

                        bool isHovered   = 
                            VisualisationUtility::DistanceToLineSegment(mousePos, lineStart, lineEnd) <
                            minLineThicknessValue + 2.0f;
                        bool isClicked   = isHovered && ImGui::IsMouseClicked(0);

                        if (isHovered)
                            hoveredWeightIndex = neuronIndex + numNeuronsPrev + prevNeuronIndex;
                        if (isClicked)
                            clickedWeightIndex = hoveredWeightIndex;

                        ImColor lineColour = ImColor(
                            isHovered ? ImVec4(255, 255, 0, 255) 
                                      : VisualisationUtility::GetWeightColor(weight)
                        );
                        ImGui::GetWindowDrawList()->AddLine(lineStart, lineEnd, lineColour, thickness);

                        if (drawWeights || isHovered || mouseHoveringNeuron)
                        {
                            lineInfos.emplace_back(lineStart, lineEnd, weight, isHovered);
                        }
                    }
                }
            }
        }
    }

    // Draw weight text near lines
    for (const LineInfo& info : lineInfos)
    {
        VisualisationUtility::DrawWeightText(info.lineStart, info.lineEnd, info.weight);
    }

    ImGui::End();

    // A second window for toggles and customization
    ImGui::Begin("Visualization Customization");

    ImGui::SliderFloat("Min Circle Size",       &minCircleSizeValue,    1.0f, 10.0f);
    ImGui::SliderFloat("Max Circle Size",       &maxCircleSizeValue,    10.0f, 100.0f);
    ImGui::SliderFloat("Circle Thickness",      &circleThicknessValue,  1.0f, 5.0f);
    ImGui::Checkbox("Draw Lines",               &drawLineConnections);
    ImGui::SliderFloat("Min Line Thickness",    &minLineThicknessValue, 1.0f, 5.0f);
    ImGui::Checkbox("Draw Weights",             &drawWeights);
    ImGui::Checkbox("Pulsating Neurons",        &activeNeuronCanPulse);

    ImGui::End();
}

// 5.3 Dataset Management Window (was NeuralNetworkStartWindow)
MyTextureData loadedInputImage;

void DatasetManagementWindow(bool* p_open, NeuralNetwork& network)
{
    if (!ImGui::Begin("Dataset Management", p_open))
    {
        ImGui::End();
        return;
    }

    // "Load Inference Image" section
    if (ImGui::Button("Open File Dialog"))
    {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".png,.jpg,.jpeg", config);
    }

    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
            filePath     = ImGuiFileDialog::Instance()->GetCurrentPath();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGui::Button("Load Inference Image") && !filePathName.empty())
    {
        // Insert Inverence loading code here. hook up to MNISTLoader
        bool ret = false;
        IM_ASSERT(ret);
    }

    // A placeholder for the next stage: training or inference
    if (ImGui::Button("Start Learning"))
    {
        // Dummy data
        std::vector<double> dummyInput(network.layers[0].numNeurons);
        for (auto& value : dummyInput)
            value = static_cast<double>(rand()) / RAND_MAX;

        std::vector<double> dummyOutput(network.layers.back().numNeurons, 0.0);
        dummyOutput[0] = 1.0;

        NeuralNetworkSubsystem::GetInstance().StartNeuralNetwork(dummyInput, dummyOutput);
    }

    ImGui::End();
}

// 5.4 Neural Network Controls Window (combines old Customisation + HyperParameter)
void NeuralNetworkControlsWindow(bool* p_open)
{
    if (!ImGui::Begin("Neural Network Controls", p_open))
    {
        ImGui::End();
        return;
    }

    // Use a TabBar to separate Architecture vs HyperParameters
    if (ImGui::BeginTabBar("##NNControlsTabs"))
    {
        // ---------- Tab 1: Architecture ----------
        if (ImGui::BeginTabItem("Architecture"))
        {
            // The old NeuralNetworkCustomisationWindow code
            static int inputLayerSize   = HyperParameters::defaultInputLayerSize;
            static int numHiddenLayers  = HyperParameters::defaultNumHiddenLayers;
            static int hiddenLayerSize  = HyperParameters::defaultHiddenLayerSize;
            static int outputLayerSize  = HyperParameters::defaultOutputLayerSize;

            ImGui::Text("Input Layer");
            ImGui::InputInt("Input Size", &inputLayerSize);
            if (inputLayerSize < 1) inputLayerSize = 1;
            ImGui::Spacing();

            ImGui::Text("Hidden Layers");
            ImGui::InputInt("Number of Hidden Layers", &numHiddenLayers);
            if (numHiddenLayers < 0) numHiddenLayers = 1;

            ImGui::InputInt("Hidden Layer Size", &hiddenLayerSize);
            if (hiddenLayerSize < 1) hiddenLayerSize = 1;
            ImGui::Spacing();

            ImGui::Text("Output Layer");
            ImGui::InputInt("Number of Outputs", &outputLayerSize);
            if (outputLayerSize < 1) outputLayerSize = 1;

            // Activation
            static int activationElem = sigmoid; // or 0
            const char* activationElemsNames[] = { "Sigmoid", "Sigmoid Derivative", "ReLU" };
            const char* activationElemName = (activationElem >= 0 && activationElem < Activation_Count) 
                                                ? activationElemsNames[activationElem] : "Unknown";
            ImGui::SliderInt("Activation Function", &activationElem, 0, Activation_Count - 1, activationElemName);

            // Cost
            static int costElem = meanSquaredError;
            const char* costElemsNames[] = { "Mean Squared Error", "Cross Entropy" };
            const char* costElemName = (costElem >= 0 && costElem < cost_Count) 
                                           ? costElemsNames[costElem] : "Unknown";
            ImGui::SliderInt("Cost Function", &costElem, 0, cost_Count - 1, costElemName);

            // Create button
            if (ImGui::Button("Create Neural Network"))
            {
                ActivationType actType;
                switch (activationElem)
                {
                case 0:  actType = sigmoid;            break;  // "Sigmoid"
                case 1:  actType = sigmoidDerivative;  break;  // "Sigmoid Derivative"
                case 2:  actType = ReLU;               break;  // "ReLU"
                default: actType = sigmoid;            break;
                }

                CostType costType;
                switch (costElem)
                {
                case 0:  costType = meanSquaredError;  break;
                case 1:  costType = crossEntropy;      break;
                default: costType = meanSquaredError;  break;
                }

                NeuralNetworkSubsystem& NN = NeuralNetworkSubsystem::GetInstance();
                NN.InitNeuralNetwork(actType, costType, 
                                     inputLayerSize, numHiddenLayers, hiddenLayerSize, outputLayerSize);
                // Visualization callback
                NN.SetVisualizationCallback([](const NeuralNetwork& net){
                    // By toggling this to true, we ensure the panel is shown
                    showVisualizationPanelWindow = true;
                });

                // Show the Visualization Panel & Dataset Management by default once created
                showVisualizationPanelWindow  = true;
                showDatasetManagementWindow   = true;
            }

            ImGui::EndTabItem();
        }

        // ---------- Tab 2: HyperParameters ----------
        if (ImGui::BeginTabItem("HyperParameters"))
        {
            // The old HyperParameterWindow code
            ImGui::Text("Training Settings");
            ImGui::InputFloat("Learning Rate", &HyperParameters::learningRate, 0.001f);
            ImGui::InputInt("Batch Size",      &HyperParameters::batchSize);
            ImGui::InputInt("Epochs",          &HyperParameters::epochs);
            ImGui::InputDouble("Momentum",     &HyperParameters::momentum, 0.1, 0.2, "%.1f");
            ImGui::InputDouble("Weight Decay", &HyperParameters::weightDecay,0.001, 0.002, "%.3f");

            ImGui::Text("Dropout");
            ImGui::Checkbox("Use Dropout", &HyperParameters::useDropoutRate);
            if (HyperParameters::useDropoutRate)
            {
                ImGui::SliderFloat("Dropout Rate", &HyperParameters::dropoutRate, 0.0f, 1.0f);
            }

            // Reset
            if (ImGui::Button("Reset HyperParameters"))
            {
                HyperParameters::ResetHyperParameters();
            }

            // Sanity checks
            if (HyperParameters::learningRate < 0)   HyperParameters::learningRate = 0;
            if (HyperParameters::batchSize   < 0)   HyperParameters::batchSize   = 0;
            if (HyperParameters::epochs      < 0)   HyperParameters::epochs      = 0;
            if (HyperParameters::momentum    < 0)   HyperParameters::momentum    = 0;
            if (HyperParameters::weightDecay < 0)   HyperParameters::weightDecay = 0;
            if (HyperParameters::dropoutRate < 0)   HyperParameters::dropoutRate = 0;

            ImGui::EndTabItem();
        }
    }
    ImGui::EndTabBar();
    ImGui::End();
}

// 5.5 Function that calls all default windows
void DefaultWindows()
{
    // 1) Dataset Management
    if (showDatasetManagementWindow)
    {
        DatasetManagementWindow(&showDatasetManagementWindow, 
                                NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork());
    }

    // 2) Neural Network Controls
    if (showNeuralNetworkControlsWindow)
    {
        NeuralNetworkControlsWindow(&showNeuralNetworkControlsWindow);
    }

    // 3) Visualization Panel
    if (showVisualizationPanelWindow)
    {
        VisualizationPanelWindow(&showVisualizationPanelWindow, 
                                 NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork());
    }

    // 4) Advanced Editing
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
    uint64_t object, size_t location, int32_t messageCode,
    const char* pLayerPrefix, const char* pMessage, void* /*pUserData*/)
{
    fprintf(stderr, "[vulkan] Debug report from ObjectType: %i\nMessage: %s\n\n", 
            objectType, pMessage);
    return VK_FALSE;
}
#endif

static bool IsExtensionAvailable(const ImVector<VkExtensionProperties>& properties, const char* extension)
{
    for (const VkExtensionProperties& p : properties)
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
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
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

        uint32_t properties_count;
        ImVector<VkExtensionProperties> properties;
        vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.Data);
        check_vk_result(err);

        if (IsExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
        {
            instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

#ifdef IMGUI_VULKAN_DEBUG_REPORT
        const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
        create_info.enabledLayerCount   = 1;
        create_info.ppEnabledLayerNames = layers;
        instance_extensions.push_back("VK_EXT_debug_report");
#endif

        create_info.enabledExtensionCount   = (uint32_t)instance_extensions.Size;
        create_info.ppEnabledExtensionNames = instance_extensions.Data;
        err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
        check_vk_result(err);

#ifdef IMGUI_VULKAN_DEBUG_REPORT
        auto vkCreateDebugReportCallbackEXT = 
            (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkCreateDebugReportCallbackEXT");
        IM_ASSERT(vkCreateDebugReportCallbackEXT != nullptr);
        VkDebugReportCallbackCreateInfoEXT debug_report_ci = {};
        debug_report_ci.sType       = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        debug_report_ci.flags       = 
            VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | 
            VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        debug_report_ci.pfnCallback = debug_report;
        debug_report_ci.pUserData   = nullptr;
        err = vkCreateDebugReportCallbackEXT(g_Instance, &debug_report_ci, g_Allocator, &g_DebugReport);
        check_vk_result(err);
#endif
    }

    // [2] Select Physical Device
    g_PhysicalDevice = SetupVulkan_SelectPhysicalDevice();

    // [3] Select graphics queue family
    {
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, nullptr);
        VkQueueFamilyProperties* queues = 
            (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * count);
        vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, queues);
        for (uint32_t i = 0; i < count; i++)
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                g_QueueFamily = i;
                break;
            }
        free(queues);
        IM_ASSERT(g_QueueFamily != (uint32_t)-1);
    }

    // [4] Create Logical Device
    {
        ImVector<const char*> device_extensions;
        device_extensions.push_back("VK_KHR_swapchain");

        uint32_t properties_count;
        ImVector<VkExtensionProperties> properties;
        vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, properties.Data);

#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME))
            device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

        const float queue_priority[] = { 1.0f };
        VkDeviceQueueCreateInfo queue_info[1] = {};
        queue_info[0].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[0].queueFamilyIndex = g_QueueFamily;
        queue_info[0].queueCount       = 1;
        queue_info[0].pQueuePriorities = queue_priority;

        VkDeviceCreateInfo create_info = {};
        create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount    = 1;
        create_info.pQueueCreateInfos       = queue_info;
        create_info.enabledExtensionCount   = (uint32_t)device_extensions.Size;
        create_info.ppEnabledExtensionNames = device_extensions.Data;

        err = vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
        check_vk_result(err);
        vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
    }

    // [5] Create Descriptor Pool
    {
        VkDescriptorPoolSize pool_sizes[] =
        {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets       = 1;
        pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
        pool_info.pPoolSizes    = pool_sizes;
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
        fprintf(stderr, "Error no WSI support on physical device 0\n");
        exit(-1);
    }

    // Surface format
    const VkFormat requestSurfaceImageFormat[] = {
        VK_FORMAT_B8G8R8A8_UNORM, 
        VK_FORMAT_R8G8B8A8_UNORM, 
        VK_FORMAT_B8G8R8_UNORM, 
        VK_FORMAT_R8G8B8_UNORM
    };
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        g_PhysicalDevice, wd->Surface, requestSurfaceImageFormat,
        (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

    // Present Mode
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
        g_PhysicalDevice, wd->Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));

    IM_ASSERT(g_MinImageCount >= 2);
    ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, 
                                           wd, g_QueueFamily, g_Allocator, 
                                           width, height, g_MinImageCount);
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
        (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkDestroyDebugReportCallbackEXT");
    vkDestroyDebugReportCallbackEXT(g_Instance, g_DebugReport, g_Allocator);
#endif
    vkDestroyDevice(g_Device, g_Allocator);
    vkDestroyInstance(g_Instance, g_Allocator);
}

static void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
{
    VkResult err;
    VkSemaphore image_acquired_semaphore  = 
        wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore = 
        wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX, 
                                image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);

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
        info.sType            = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass       = wd->RenderPass;
        info.framebuffer      = fd->Framebuffer;
        info.renderArea.extent.width  = wd->Width;
        info.renderArea.extent.height = wd->Height;
        info.clearValueCount  = 1;
        info.pClearValues     = &wd->ClearValue;
        vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    // Record dear imgui primitives
    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

    vkCmdEndRenderPass(fd->CommandBuffer);
    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit_info        = {};
        submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount   = 1;
        submit_info.pWaitSemaphores     = &image_acquired_semaphore;
        submit_info.pWaitDstStageMask   = &wait_stage;
        submit_info.commandBufferCount   = 1;
        submit_info.pCommandBuffers      = &fd->CommandBuffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores    = &render_complete_semaphore;

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
    VkSemaphore render_complete_semaphore = 
        wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores    = &render_complete_semaphore;
    info.swapchainCount     = 1;
    info.pSwapchains        = &wd->Swapchain;
    info.pImageIndices      = &wd->FrameIndex;
    VkResult err            = vkQueuePresentKHR(g_Queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    {
        g_SwapChainRebuild = true;
        return;
    }
    check_vk_result(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount;
}

//--------------------------------------------------------------------
// [SECTION 7] Main() + Callbacks
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

// Main code
int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Create window with Vulkan context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Mike's Neural Network Thing", nullptr, nullptr);
    if (!glfwVulkanSupported())
    {
        printf("GLFW: Vulkan Not Supported\n");
        return 1;
    }

    // Collect required extensions
    ImVector<const char*> extensions;
    uint32_t extensions_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&extensions_count);
    for (uint32_t i = 0; i < extensions_count; i++)
        extensions.push_back(glfw_extensions[i]);

    // Set up Vulkan & Window Surface
    SetupVulkan(extensions);

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
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Keyboard
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Gamepad
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Multi-Viewport
    ImGui::StyleColorsDark();

    // Adjust style for multi-viewport
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding              = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance       = g_Instance;
    init_info.PhysicalDevice = g_PhysicalDevice;
    init_info.Device         = g_Device;
    init_info.QueueFamily    = g_QueueFamily;
    init_info.Queue          = g_Queue;
    init_info.PipelineCache  = g_PipelineCache;
    init_info.DescriptorPool = g_DescriptorPool;
    init_info.Subpass        = 0;
    init_info.MinImageCount  = g_MinImageCount;
    init_info.ImageCount     = wd->ImageCount;
    init_info.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator      = g_Allocator;
    init_info.CheckVkResultFn= check_vk_result;
    ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);

    // Load Fonts (optional)
    // io.Fonts->AddFontDefault();
    // [Optional] e.g. io.Fonts->AddFontFromFileTTF("path_to_font.ttf", 16.0f);

    // Upload ImGui Fonts to GPU
    {
        // Use any command queue
        VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
        VkCommandBuffer command_buffer;

        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool        = command_pool;
        alloc_info.commandBufferCount = 1;

        err = vkAllocateCommandBuffers(g_Device, &alloc_info, &command_buffer);
        check_vk_result(err);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(command_buffer, &begin_info);
        check_vk_result(err);


        VkSubmitInfo end_info = {};
        end_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        end_info.commandBufferCount   = 1;
        end_info.pCommandBuffers      = &command_buffer;
        err = vkEndCommandBuffer(command_buffer);
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

        if (g_SwapChainRebuild)
        {
            glfwGetFramebufferSize(window, &w, &h);
            if (w > 0 && h > 0)
            {
                ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
                ImGui_ImplVulkanH_CreateOrResizeWindow(
                    g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData, 
                    g_QueueFamily, g_Allocator, w, h, g_MinImageCount);
                g_MainWindowData.FrameIndex = 0;
                g_SwapChainRebuild = false;
            }
        }

        // Start the ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Draw the default windows
        DefaultWindows();

        // Render
        ImGui::Render();
        ImDrawData* main_draw_data = ImGui::GetDrawData();
        const bool main_is_minimized = 
            (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);

        wd->ClearValue.color.float32[0] = 0.0f;
        wd->ClearValue.color.float32[1] = 0.0f;
        wd->ClearValue.color.float32[2] = 0.0f;
        wd->ClearValue.color.float32[3] = 1.0f;

        if (!main_is_minimized)
            FrameRender(wd, main_draw_data);

        // Render additional Platform Windows
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        // Present
        if (!main_is_minimized)
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
