#include "cstdio"          // printf, fprintf
#include "cstdlib"         // abort
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "Cost.h"
#include "HyperParameters.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkSubsystem.h"
#include "string"
#include "GLFW/glfw3.h"
#include "vulkan/vulkan.h"

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

//#define IMGUI_UNLIMITED_FRAME_RATE
#ifdef _DEBUG
#define IMGUI_VULKAN_DEBUG_REPORT
#endif

// Data
static VkAllocationCallbacks* g_Allocator = nullptr;
static VkInstance               g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice         g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice                 g_Device = VK_NULL_HANDLE;
static uint32_t                 g_QueueFamily = (uint32_t)-1;
static VkQueue                  g_Queue = VK_NULL_HANDLE;
static VkDebugReportCallbackEXT g_DebugReport = VK_NULL_HANDLE;
static VkPipelineCache          g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool         g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static int                      g_MinImageCount = 2;
static bool                     g_SwapChainRebuild = false;


// Windows

bool showNeuralNetworkCustomisationWindow = true;
bool showHyperParameterWindow = true;
bool showWeightsAndBiasesWindow = true;
bool showNeuralNetworkWindow = false;
bool showNeuralNetworkStartWindow  = false;

// Neural Network Customisation Window
int inputLayerSize = HyperParameters::defaultInputLayerSize;
int numHiddenLayers = HyperParameters::defaultNumHiddenLayers;
int hiddenLayerSize = HyperParameters::defaultHiddenLayerSize;
int outputLayerSize = HyperParameters::defaultOutputLayerSize;

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}
static void check_vk_result(VkResult err)
{
	if (err == 0)
		return;
	fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
	if (err < 0)
		abort();
}

#ifdef IMGUI_VULKAN_DEBUG_REPORT
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
	(void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; // Unused arguments
	fprintf(stderr, "[vulkan] Debug report from ObjectType: %i\nMessage: %s\n\n", objectType, pMessage);
	return VK_FALSE;
}
#endif // IMGUI_VULKAN_DEBUG_REPORT

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

	// If a number >1 of GPUs got reported, find discrete GPU if present, or use first one available. This covers
	// most common cases (multi-gpu/integrated+dedicated graphics). Handling more complicated setups (multiple
	// dedicated GPUs) is out of scope of this sample.
	for (VkPhysicalDevice& device : gpus)
	{
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(device, &properties);
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			return device;
	}

	// Use first GPU (Integrated) is a Discrete one is not available.
	if (gpu_count > 0)
		return gpus[0];
	return VK_NULL_HANDLE;
}

static void SetupVulkan(ImVector<const char*> instance_extensions)
{
	VkResult err;

	// Create Vulkan Instance
	{
		VkInstanceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

		// Enumerate available extensions
		uint32_t properties_count;
		ImVector<VkExtensionProperties> properties;
		vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
		properties.resize(properties_count);
		err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.Data);
		check_vk_result(err);

		// Enable required extensions
		if (IsExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
			instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
		if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
		{
			instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
			create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
		}
#endif

		// Enabling validation layers
#ifdef IMGUI_VULKAN_DEBUG_REPORT
		const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
		create_info.enabledLayerCount = 1;
		create_info.ppEnabledLayerNames = layers;
		instance_extensions.push_back("VK_EXT_debug_report");
#endif

		// Create Vulkan Instance
		create_info.enabledExtensionCount = (uint32_t)instance_extensions.Size;
		create_info.ppEnabledExtensionNames = instance_extensions.Data;
		err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
		check_vk_result(err);

		// Setup the debug report callback
#ifdef IMGUI_VULKAN_DEBUG_REPORT
		auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkCreateDebugReportCallbackEXT");
		IM_ASSERT(vkCreateDebugReportCallbackEXT != nullptr);
		VkDebugReportCallbackCreateInfoEXT debug_report_ci = {};
		debug_report_ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		debug_report_ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
		debug_report_ci.pfnCallback = debug_report;
		debug_report_ci.pUserData = nullptr;
		err = vkCreateDebugReportCallbackEXT(g_Instance, &debug_report_ci, g_Allocator, &g_DebugReport);
		check_vk_result(err);
#endif
	}

	// Select Physical Device (GPU)
	g_PhysicalDevice = SetupVulkan_SelectPhysicalDevice();

	// Select graphics queue family
	{
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, nullptr);
		VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * count);
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

	// Create Logical Device (with 1 queue)
	{
		ImVector<const char*> device_extensions;
		device_extensions.push_back("VK_KHR_swapchain");

		// Enumerate physical device extension
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
		queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[0].queueFamilyIndex = g_QueueFamily;
		queue_info[0].queueCount = 1;
		queue_info[0].pQueuePriorities = queue_priority;
		VkDeviceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]);
		create_info.pQueueCreateInfos = queue_info;
		create_info.enabledExtensionCount = (uint32_t)device_extensions.Size;
		create_info.ppEnabledExtensionNames = device_extensions.Data;
		err = vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
		check_vk_result(err);
		vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
	}

	// Create Descriptor Pool
	// The example only requires a single combined image sampler descriptor for the font image and only uses one descriptor set (for that)
	// If you wish to load e.g. additional textures you may need to alter pools sizes.
	{
		VkDescriptorPoolSize pool_sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
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

// All the ImGui_ImplVulkanH_XXX structures/functions are optional helpers used by the demo.
// Your real engine/app may not use them.
static void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height)
{
	wd->Surface = surface;

	// Check for WSI support
	VkBool32 res;
	vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_QueueFamily, wd->Surface, &res);
	if (res != VK_TRUE)
	{
		fprintf(stderr, "Error no WSI support on physical device 0\n");
		exit(-1);
	}

	// Select Surface Format
	const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
	const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
	wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(g_PhysicalDevice, wd->Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

	// Select Present Mode
#ifdef IMGUI_UNLIMITED_FRAME_RATE
	VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
#else
	VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
#endif
	wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(g_PhysicalDevice, wd->Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));
	//printf("[vulkan] Selected PresentMode = %d\n", wd->PresentMode);

	// Create SwapChain, RenderPass, Framebuffer, etc.
	IM_ASSERT(g_MinImageCount >= 2);
	ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, wd, g_QueueFamily, g_Allocator, width, height, g_MinImageCount);
}

static void CleanupVulkan()
{
	vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

#ifdef IMGUI_VULKAN_DEBUG_REPORT
	// Remove the debug report callback
	auto vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkDestroyDebugReportCallbackEXT");
	vkDestroyDebugReportCallbackEXT(g_Instance, g_DebugReport, g_Allocator);
#endif // IMGUI_VULKAN_DEBUG_REPORT

	vkDestroyDevice(g_Device, g_Allocator);
	vkDestroyInstance(g_Instance, g_Allocator);
}

static void CleanupVulkanWindow()
{
	ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData, g_Allocator);
}

static void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
{
	VkResult err;

	VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
	VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
	err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
	if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
	{
		g_SwapChainRebuild = true;
		return;
	}
	check_vk_result(err);

	ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
	{
		err = vkWaitForFences(g_Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);    // wait indefinitely instead of periodically checking
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

	// Record dear imgui primitives into command buffer
	ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

	// Submit command buffer
	vkCmdEndRenderPass(fd->CommandBuffer);
	{
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &image_acquired_semaphore;
		info.pWaitDstStageMask = &wait_stage;
		info.commandBufferCount = 1;
		info.pCommandBuffers = &fd->CommandBuffer;
		info.signalSemaphoreCount = 1;
		info.pSignalSemaphores = &render_complete_semaphore;

		err = vkEndCommandBuffer(fd->CommandBuffer);
		check_vk_result(err);
		err = vkQueueSubmit(g_Queue, 1, &info, fd->Fence);
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
	wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount; // Now we can use the next set of semaphores
}

void WeightsAndBiasesWindow(bool* p_open, NeuralNetwork& network)
{
	
	// ImGui window setup
	if (!ImGui::Begin("Edit Neural Network", p_open)) {
		ImGui::End();
		return;
	}

	// Iterate through each layer in the network
	for (size_t layerIndex = 0; layerIndex < network.layers.size(); ++layerIndex) {
		const Layer& layer = network.layers[layerIndex];
		ImGui::Text("Layer %zu", layerIndex);

		// Display biases for the current layer
		ImGui::Text("Biases:");
		for (size_t neuronIndex = 0; neuronIndex < layer.biases.size(); ++neuronIndex) {

			float biasAsFloat = static_cast<float>(network.layers[layerIndex].biases[neuronIndex]);
			ImGui::SliderFloat(
				("Bias##" + std::to_string(layerIndex) + "_" + std::to_string(neuronIndex)).c_str(),
				&biasAsFloat,-2.0f, 2.0f,"%.1f"
			);
			network.layers[layerIndex].biases[neuronIndex] = static_cast<double>(biasAsFloat);
		}

		// Display weights for the current layer
		ImGui::Text("Weights:");
		if (!layer.weights.empty()) {
			for (size_t neuronIndex = 0; neuronIndex < layer.weights.size(); ++neuronIndex) {
				for (size_t weightIndex = 0; weightIndex < layer.weights[neuronIndex].size(); ++weightIndex) {
					
					float weightAsFloat = static_cast<float>(network.layers[layerIndex].weights[neuronIndex][weightIndex]);

					ImGui::SliderFloat(
						("Weight##" + std::to_string(layerIndex) + "_" + std::to_string(neuronIndex) + "_" + std::to_string(weightIndex)).c_str(),
						&weightAsFloat,-0.01f, 0.01f,"%.3f"
					);
					network.layers[layerIndex].weights[neuronIndex][weightIndex] = static_cast<double>(weightAsFloat);
				}
			}
		}

		ImGui::Separator(); // Separate layers visually
	}

	ImGui::End();
}

void NeuralNetworkCustomisationWindow(bool* p_open)
{
	// ImGui window setup
	if (!ImGui::Begin("Neural Network Settings", p_open)) {
		ImGui::End();
		return;
	}

	// Input layer settings
	ImGui::Text("Input Layer");
	ImGui::InputInt("Input Size", &inputLayerSize);

	if (inputLayerSize < 1)
	{
		inputLayerSize = 1;
	}

	ImGui::Spacing();

	// Hidden layer settings
	ImGui::Text("Hidden Layers");
	ImGui::InputInt("Number of Hidden Layers", &numHiddenLayers);

	if (numHiddenLayers < 0)
	{
		numHiddenLayers = 1;
	}

	ImGui::InputInt("Hidden Layer Size", &hiddenLayerSize);

	if (hiddenLayerSize < 1)
	{
		hiddenLayerSize = 1;
	}

	ImGui::Spacing();

	// Output layer settings
	ImGui::Text("Output Layer");
	ImGui::InputInt("Number of Output Layers", &outputLayerSize);

	if (outputLayerSize < 1)
	{
		outputLayerSize = 1;
	}

	static int activationElem = sigmoid;
	const char* activationElemsNames[] = { "Sigmoid", "Sigmoid Derivative", "ReLU" };
	const char* activationElemName = (activationElem >= 0 && activationElem < Activation_Count) ? activationElemsNames[activationElem] : "Unknown";
	ImGui::SliderInt("Activation Function", &activationElem, 0, Activation_Count - 1, activationElemName);

	static int costElem = meanSquaredError;
	const char* costElemsNames[] = { "Mean Squared Error", "Cross Entropy"};
	const char* costElemName = (costElem >= 0 && costElem < cost_Count) ? costElemsNames[costElem] : "Unknown";
	ImGui::SliderInt("Cost Function", &costElem, 0, cost_Count - 1, costElemName);

	constexpr double defaultBias = 0.1;
	constexpr double defaultWeights = 0;

	// Reset button action
	if (ImGui::Button("Create NeuralNetwork")) {

		ActivationType activation;
		switch (activationElem)
		{
		case 1:

			activation = sigmoid;
			break;

		case 2:

			activation = sigmoidDerivative;
			break;

		case 3:

			activation = ReLU;
			break;

		default:
			activation = sigmoid;
			break;
		}

		CostType cost;
		switch (costElem)
		{
		case 1:

			cost = meanSquaredError;
			break;

		case 2:

			cost = crossEntropy;
			break;

		default:
			cost = meanSquaredError;
			break;
		}

		NeuralNetworkSubsystem::GetInstance().InitNeuralNetwork(activation, cost, inputLayerSize, numHiddenLayers,
		                                                        hiddenLayerSize, outputLayerSize);

		showNeuralNetworkWindow = true;
		showNeuralNetworkStartWindow = true;
	}

	// Reset button action
	if (ImGui::Button("Delete NeuralNetwork"))
	{
		NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork() = NeuralNetwork();
		showNeuralNetworkWindow = false;
	}
	ImGui::End();

}

void HyperParameterWindow(bool* p_open) {

	// ImGui window setup
	if (!ImGui::Begin("HyperParameter Settings", p_open)) {
		ImGui::End();
		return;
	}

	//ImGui::Spacing();

	// Training settings
	ImGui::Text("Training");
	ImGui::InputFloat("Learning Rate", &HyperParameters::learningRate, 0.001f);
	ImGui::InputInt("Batch Size", &HyperParameters::batchSize);
	ImGui::InputInt("Epochs", &HyperParameters::epochs);
	ImGui::InputDouble("Momentum", &HyperParameters::momentum, 0.1, 0.2, "%.1f");
	ImGui::InputDouble("Weight Decay", &HyperParameters::weightDecay,0.001, 0.002, "%.3f");
	ImGui::Text("Dropout");
	ImGui::Checkbox("Use Dropout", &HyperParameters::useDropoutRate);
	if (HyperParameters::useDropoutRate)
	{
		ImGui::SliderFloat("Dropout Rate", &HyperParameters::dropoutRate, 0.0f, 1.0f);
	}

	// Reset button action
	if (ImGui::Button("Reset HyperParameters")) {
		HyperParameters::ResetHyperParameters();
	}

	if (HyperParameters::learningRate < 0)
	{
		HyperParameters::learningRate = 0;
	}

	if (HyperParameters::batchSize < 0)
	{
		HyperParameters::batchSize = 0;
	}

	if (HyperParameters::epochs < 0)
	{
		HyperParameters::epochs = 0;
	}

	if (HyperParameters::momentum < 0)
	{
		HyperParameters::momentum = 0;
	}

	if (HyperParameters::weightDecay < 0)
	{
		HyperParameters::weightDecay = 0;
	}

	if (HyperParameters::dropoutRate < 0)
	{
		HyperParameters::dropoutRate = 0;
	}

	ImGui::End();
}


float minCircleSizeValue = 5.0f;
float maxCircleSizeValue = 50.0f;
float circleThicknessValue = 1.0f;
bool drawLineConnections = true;

void NeuralNetworkWindow(bool* p_open, const NeuralNetwork& network) {
	if (!ImGui::Begin("Neural Network Visualization", p_open)) {
		ImGui::End();
		return;
	}

	const ImVec2 windowSize = ImGui::GetWindowSize();
	const ImVec2 windowPos = ImGui::GetWindowPos();
	const float windowScrollY = ImGui::GetScrollY();

	// Calculate maximum allowable circle size based on the window and network size
	const float maxCircleSize = std::min(std::min(windowSize.x, windowSize.y) / (network.layers.size() * 2.0f), maxCircleSizeValue);

	// Calculate spacing between layers to fit them within the window
	const float layerSpacing = windowSize.x / (network.layers.size() + 1);

	// Loop through layers
	for (int layerIndex = 0; layerIndex < network.layers.size(); ++layerIndex) {
		const Layer& layer = network.layers[layerIndex];
		const float layerPosX = (layerIndex + 1) * layerSpacing;

		const int numNeurons = layer.numNeurons > 0 ? layer.numNeurons : 1; // Ensure at least one neuron is displayed

		// Calculate spacing between neurons within a layer
		const float neuronSpacing = windowSize.y / (numNeurons + 1);

		// Loop through neurons in the layer
		for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
			const float circleSize = std::min(maxCircleSize, neuronSpacing * 0.5f);

			// Calculate position for each neuron
			const float posX = layerPosX;
			const float posY = (neuronIndex + 1) * neuronSpacing;

			ImGui::SetCursorPos(ImVec2(posX - circleSize, posY - circleSize));
			ImGui::BeginGroup();
			ImGui::PushID(layerIndex * 1000 + neuronIndex);

			ImGui::GetWindowDrawList()->AddCircle(
				ImVec2(ImGui::GetCursorScreenPos().x + circleSize, ImGui::GetCursorScreenPos().y + circleSize),
				circleSize,
				ImColor(255, 255, 255),
				12,
				1.0f // Circle thickness
			);

			char buffer[32];
			std::snprintf(buffer, sizeof(buffer), "%.1f", layer.neurons[neuronIndex].ActivationValue);
			const ImVec2 textSize = ImGui::CalcTextSize(buffer);
			const ImVec2 textPos = ImVec2(posX - textSize.x * 0.5f, posY - textSize.y * 0.5f);
			ImGui::SetCursorPos(textPos);
			ImGui::Text("%s", buffer);

			ImGui::PopID();
			ImGui::EndGroup();

			if (layerIndex > 0 && drawLineConnections) {
				const float prevLayerPosX = layerPosX - layerSpacing;
				const int numNeuronsPrevLayer = network.layers[layerIndex - 1].numNeurons > 0 ? network.layers[layerIndex - 1].numNeurons : 1;

				const float neuronSpacingPrevLayer = windowSize.y / (numNeuronsPrevLayer + 1);
				const float prevCircleSize = std::min(maxCircleSize, neuronSpacingPrevLayer * 0.5f);

				for (int prevNeuronIndex = 0; prevNeuronIndex < numNeuronsPrevLayer; ++prevNeuronIndex) {
					const float prevPosY = (prevNeuronIndex + 1) * neuronSpacingPrevLayer;

					ImVec2 lineStart = ImVec2(prevLayerPosX + prevCircleSize + windowPos.x, prevPosY + windowPos.y - windowScrollY);
					ImVec2 lineEnd = ImVec2(posX - circleSize + windowPos.x, posY + windowPos.y - windowScrollY);

					ImGui::GetWindowDrawList()->AddLine(
						lineStart,
						lineEnd,
						ImColor(255, 255, 255),
						1.0f // Line thickness
					);
				}
			}
		}
	}

	ImGui::End();

	ImGui::Begin("Visual Customization");

	ImGui::SliderFloat("Min Circle Size", &minCircleSizeValue, 1.0f, 10.0f);
	ImGui::SliderFloat("Max Circle Size", &maxCircleSizeValue, 10.0f, 100.0f);
	ImGui::SliderFloat("Circle Thickness", &circleThicknessValue, 1.0f, 5.0f);
	ImGui::Checkbox("Draw Lines", &drawLineConnections);


	ImGui::End();
}

void NeuralNetworkStartWindow(bool* p_open, NeuralNetwork& network)
{
	if (!ImGui::Begin("Neural Network Begin Test", p_open)) {
		ImGui::End();
		return;
	}

	for (int i = 0; i < network.layers[0].numNeurons; i++)
	{
		ImGui::InputDouble(("Input Data for Neuron:##" + std::to_string(i) + "_" + std::to_string(i)).c_str(), &network.layers[0].neurons[i].ActivationValue, 0.01, 0.1, "%.2f");
	}

	if (ImGui::Button("Start Learning"))
	{
		
		std::vector<double> input(network.layers[0].numNeurons);

		for (int i = 0; i < network.layers[0].numNeurons; i++)
		{
			input[i] = network.layers[0].neurons[i].ActivationValue;
		}
		NeuralNetworkSubsystem::GetInstance().StartNeuralNetwork(input);
	}


	ImGui::End();
}

void DefaultWindows()
{
	if (showNeuralNetworkCustomisationWindow)
	{
		NeuralNetworkCustomisationWindow(&showNeuralNetworkCustomisationWindow);
	}

	if (showHyperParameterWindow)
	{
		HyperParameterWindow(&showHyperParameterWindow);
	}

	if (showNeuralNetworkWindow)
	{
		NeuralNetworkWindow(&showNeuralNetworkWindow, NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork());
	}

	if (showNeuralNetworkStartWindow)
	{
		NeuralNetworkStartWindow(&showNeuralNetworkStartWindow, NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork());
	}

	if (showWeightsAndBiasesWindow)
	{
		WeightsAndBiasesWindow(&showWeightsAndBiasesWindow, NeuralNetworkSubsystem::GetInstance().GetNeuralNetwork());
	}
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

	ImVector<const char*> extensions;
	uint32_t extensions_count = 0;
	const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&extensions_count);
	for (uint32_t i = 0; i < extensions_count; i++)
		extensions.push_back(glfw_extensions[i]);
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
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
	//io.ConfigViewportsNoAutoMerge = true;
	//io.ConfigViewportsNoTaskBarIcon = true;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
	ImGuiStyle& style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
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

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
	// - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
	// - Read 'docs/FONTS.md' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();
	//io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != nullptr);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		// Poll and handle events (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		glfwPollEvents();

		// Resize swap chain?
		if (g_SwapChainRebuild)
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			if (width > 0 && height > 0)
			{
				ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
				ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData, g_QueueFamily, g_Allocator, width, height, g_MinImageCount);
				g_MainWindowData.FrameIndex = 0;
				g_SwapChainRebuild = false;
			}
		}

		// Start the Dear ImGui frame
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		DefaultWindows();

		// Rendering
		ImGui::Render();
		ImDrawData* main_draw_data = ImGui::GetDrawData();
		const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);
		wd->ClearValue.color.float32[0] = 0 * 1;
		wd->ClearValue.color.float32[1] = 0 * 1;
		wd->ClearValue.color.float32[2] = 0 * 1;
		wd->ClearValue.color.float32[3] = 1;
		if (!main_is_minimized)
			FrameRender(wd, main_draw_data);

		// Update and Render additional Platform Windows
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}

		// Present Main Platform Window
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
