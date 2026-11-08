/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <vcl/graphics/vulkan/platform.h>

// C++ standard library
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <set>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Vulkan {
	void initializeVulkan(std::vector<VkLayerProperties>& layers, std::vector<VkExtensionProperties>& availableExtensions, std::multimap<std::string, std::string>& extensionsPerLayer)
	{
		VkResult res;

		// Enumerate the layers
		uint32_t nr_layers = 0;
		res = vkEnumerateInstanceLayerProperties(&nr_layers, nullptr);
		if (res != VkResult::VK_SUCCESS)
			throw "";

		layers.resize(nr_layers);
		res = vkEnumerateInstanceLayerProperties(&nr_layers, layers.data());
		if (res != VkResult::VK_SUCCESS)
			throw "";

		// Enumerate the extensions
		{
			uint32_t nr_extensions = 0;
			res = vkEnumerateInstanceExtensionProperties("", &nr_extensions, nullptr);
			if (res != VkResult::VK_SUCCESS)
				throw "";

			availableExtensions.clear();
			availableExtensions.resize(nr_extensions);
			res = vkEnumerateInstanceExtensionProperties("", &nr_extensions, availableExtensions.data());
			if (res != VkResult::VK_SUCCESS)
				throw "";
		}

		std::vector<VkExtensionProperties> extensions;
		for (uint32_t l = 0; l < nr_layers; l++)
		{
			uint32_t nr_extensions = 0;
			res = vkEnumerateInstanceExtensionProperties(layers[l].layerName, &nr_extensions, nullptr);
			if (res != VkResult::VK_SUCCESS)
				throw "";

			extensions.clear();
			extensions.resize(nr_extensions);
			res = vkEnumerateInstanceExtensionProperties(layers[l].layerName, &nr_extensions, extensions.data());
			if (res != VkResult::VK_SUCCESS)
				throw "";

			for (const auto& ext : extensions)
			{
				extensionsPerLayer.emplace(std::make_pair<std::string, std::string>(layers[l].layerName, ext.extensionName));
			}
		}
	}

	VKAPI_ATTR VkBool32 VKAPI_CALL VclVulkanDebugMessage(
		VkDebugUtilsMessageSeverityFlagBitsEXT severity,
		VkDebugUtilsMessageTypeFlagsEXT type,
		const VkDebugUtilsMessengerCallbackDataEXT* data,
		void* user_data)
	{
		(void)user_data;

		// General messages are mostly loader notes (e.g. drivers it skips), keep only errors
		const bool general_only = (type & ~VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) == 0;
		if (general_only && severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
			return VK_FALSE;

		const char* severity_name = "Info";
		if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
			severity_name = "Error";
		else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
			severity_name = "Warning";

		const char* type_name = "general";
		if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
			type_name = "validation";
		else if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
			type_name = "performance";

		// The message usually starts with the ID name already, e.g. "Validation Error: [ VUID-... ]"
		std::cerr << "[Vulkan] " << severity_name << " (" << type_name << ")";
		if (data->pMessageIdName && !std::strstr(data->pMessage, data->pMessageIdName))
			std::cerr << " " << data->pMessageIdName;
		std::cerr << ": " << data->pMessage << std::endl;

		// Don't abort the call that triggered the message
		return VK_FALSE;
	}

	VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo()
	{
		VkDebugUtilsMessengerCreateInfoEXT info = {};
		info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		info.pfnUserCallback = &VclVulkanDebugMessage;
		return info;
	}

	bool contains(const std::vector<const char*>& names, const char* name)
	{
		return std::any_of(std::begin(names), std::end(names), [name](const char* n) { return std::strcmp(n, name) == 0; });
	}

	template<typename Func>
	Func getInstanceProc(VkInstance inst, const char* name)
	{
		return reinterpret_cast<Func>(vkGetInstanceProcAddr(inst, name));
	}

	Platform::Platform(stdext::span<const char*> requested_extensions)
	{
		initializeVulkan(_availableLayers, _availableExtensions, _extensionsPerLayer);

		VkResult res;

		// Instance information
		VkInstanceCreateInfo create_info;
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.pNext = nullptr;
		create_info.flags = 0;

		// Information about the application
		VkApplicationInfo application_info = {};
		application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		application_info.pNext = nullptr;
		application_info.apiVersion = VK_MAKE_VERSION(1, 3, 0);
		create_info.pApplicationInfo = &application_info;

		// Enable additional layers
		std::vector<const char*> req_layers;
#ifdef VCL_DEBUG
		// Enable standard validation layers ba default, if they are available
		if (std::find_if(std::begin(_availableLayers), std::end(_availableLayers), [](const VkLayerProperties& l) {
				return strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0;
			}) != std::end(_availableLayers))
		{
			req_layers.push_back("VK_LAYER_KHRONOS_validation");
		}
#endif // VCL_DEBUG
		create_info.enabledLayerCount = (uint32_t)req_layers.size();
		create_info.ppEnabledLayerNames = req_layers.data();

		// Enable additional extensions
		std::vector<const char*> exts(std::begin(requested_extensions), std::end(requested_extensions));

		// Route the messages of the validation layers through our callback whenever
		// the extension is available. Without layers, it costs nothing; with layers
		// enabled from the outside (VK_INSTANCE_LAYERS), messages are still reported.
		const bool debug_utils_available = std::any_of(std::begin(_availableExtensions), std::end(_availableExtensions), [](const VkExtensionProperties& e) {
			return std::strcmp(e.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
		});
		if (debug_utils_available && !contains(exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
		{
			exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		const bool use_debug_utils = contains(exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		create_info.enabledExtensionCount = static_cast<uint32_t>(exts.size());
		create_info.ppEnabledExtensionNames = exts.data();

		// Also report messages from vkCreateInstance and vkDestroyInstance
		const VkDebugUtilsMessengerCreateInfoEXT messenger_info = debugMessengerCreateInfo();
		if (use_debug_utils)
			create_info.pNext = &messenger_info;

		// Create the instance
		res = vkCreateInstance(&create_info, nullptr, &_instance);
		if (res != VkResult::VK_SUCCESS)
			throw "";

		// Enumerate the available devices
		uint32_t nr_devs = 0;
		res = vkEnumeratePhysicalDevices(_instance, &nr_devs, nullptr);
		if (res != VkResult::VK_SUCCESS || nr_devs == 0)
			throw "";

		std::vector<VkPhysicalDevice> devs(nr_devs);
		res = vkEnumeratePhysicalDevices(_instance, &nr_devs, devs.data());
		if (res != VkResult::VK_SUCCESS)
			throw "";

		// Add the devices to the platform
		for (auto dev : devs)
		{
			_devices.emplace_back(dev);
		}

		// Register the messenger for the lifetime of the instance
		if (use_debug_utils)
		{
			const auto vkCreateDebugUtilsMessengerEXT = getInstanceProc<PFN_vkCreateDebugUtilsMessengerEXT>(_instance, "vkCreateDebugUtilsMessengerEXT");
			vkDestroyDebugUtilsMessengerEXT = getInstanceProc<PFN_vkDestroyDebugUtilsMessengerEXT>(_instance, "vkDestroyDebugUtilsMessengerEXT");
			if (vkCreateDebugUtilsMessengerEXT && vkDestroyDebugUtilsMessengerEXT)
			{
				res = vkCreateDebugUtilsMessengerEXT(_instance, &messenger_info, nullptr, &_debugMessenger);
				VclCheck(res == VK_SUCCESS, "Debug messenger is created.");
			}
		}
	}

	Platform::~Platform()
	{
		_devices.clear();

		// Cleanup the debugging interface
		if (_debugMessenger)
			vkDestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);

		// Cleanup the platform
		vkDestroyInstance(_instance, nullptr);
	}

	int Platform::nrDevices() const
	{
		return static_cast<int>(_devices.size());
	}

	Device& Platform::device(int idx)
	{
		VclRequire(idx < _devices.size(), "idx is valid.");

		return _devices[idx];
	}
}}}
