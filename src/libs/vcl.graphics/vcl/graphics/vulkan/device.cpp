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
#include <vcl/graphics/vulkan/device.h>

// C++ standard library
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	Device::Device(VkPhysicalDevice dev)
	: _device(dev)
	{
		// Properties of the device
		VkPhysicalDeviceProperties dev_props;
		vkGetPhysicalDeviceProperties(dev, &dev_props);

		// Store the relevant data
		_name = dev_props.deviceName;

		// Enumerate layers and extensions
		enumerateLayersAndExtensions(_device, _availableLayers, _availableExtensions, _extensionsPerLayer);

		// Enumerate the number of queue families
		uint32_t nr_queues = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(dev, &nr_queues, nullptr);

		_queueFamilies.resize(nr_queues);
		vkGetPhysicalDeviceQueueFamilyProperties(dev, &nr_queues, _queueFamilies.data());

		// Enumerte the device features
		vkGetPhysicalDeviceFeatures(dev, &_features);

		// Query the available memory
		vkGetPhysicalDeviceMemoryProperties(dev, &_memory);
	}

	std::unique_ptr<Context> Device::createContext(gsl::span<const char*> extensions)
	{
		// Enable additional layers
		std::vector<const char*> req_layers;
#ifdef VCL_DEBUG
		// Enable standard validation layers ba default, if they are available
		if (std::find_if(std::begin(_availableLayers), std::end(_availableLayers), [](const VkLayerProperties& l)
		{
			return strcmp(l.layerName, "VK_LAYER_LUNARG_standard_validation") == 0;
		}) != std::end(_availableLayers))
		{
			req_layers.push_back("VK_LAYER_LUNARG_standard_validation");
		}
#endif // VCL_DEBUG

		// Enable additional extensions
		std::vector<const char*> req_ext(std::begin(extensions), std::end(extensions));

		return std::make_unique<Context>(this, req_layers, req_ext);
	}

	gsl::span<const VkMemoryType> Device::memoryTypes() const
	{
		return{ _memory.memoryTypes, _memory.memoryTypeCount };
	}

	uint32_t Device::getMemoryTypeIndex(uint32_t typeBits, VkFlags properties)
	{
		for (uint32_t i = 0; i < 32; i++)
		{
			if ((typeBits & 1) == 1)
			{
				if ((_memory.memoryTypes[i].propertyFlags & properties) == properties)
				{
					return i;
				}
			}
			typeBits >>= 1;
		}
		return 0xffffffff;
	}

	void Device::enumerateLayersAndExtensions
	(
		VkPhysicalDevice device,
		std::vector<VkLayerProperties>& layers,
		std::vector<VkExtensionProperties>& availableExtensions,
		std::multimap<std::string, std::string>& extensionsPerLayer
	)
	{
		VkResult res;

		// Enumerate device layers
		uint32_t nr_layers = 0;
		res = vkEnumerateDeviceLayerProperties(device, &nr_layers, nullptr);
		if (res != VkResult::VK_SUCCESS)
			throw "";

		layers.resize(nr_layers);
		res = vkEnumerateDeviceLayerProperties(device, &nr_layers, layers.data());
		if (res != VkResult::VK_SUCCESS)
			throw "";

		// Enumerate the extensions
		{
			uint32_t nr_extensions = 0;
			res = vkEnumerateDeviceExtensionProperties(device, "", &nr_extensions, nullptr);
			if (res != VkResult::VK_SUCCESS)
				throw "";

			availableExtensions.clear();
			availableExtensions.resize(nr_extensions);
			res = vkEnumerateDeviceExtensionProperties(device, "", &nr_extensions, availableExtensions.data());
			if (res != VkResult::VK_SUCCESS)
				throw "";

		}
		std::vector<VkExtensionProperties> extensions;
		for (uint32_t l = 0; l < nr_layers; l++)
		{
			uint32_t nr_extensions = 0;
			res = vkEnumerateDeviceExtensionProperties(device, layers[l].layerName, &nr_extensions, nullptr);
			if (res != VkResult::VK_SUCCESS)
				throw "";

			extensions.clear();
			extensions.resize(nr_extensions);
			res = vkEnumerateDeviceExtensionProperties(device, layers[l].layerName, &nr_extensions, extensions.data());
			if (res != VkResult::VK_SUCCESS)
				throw "";

			for (const auto& ext : extensions)
			{
				extensionsPerLayer.emplace(std::make_pair<std::string, std::string>(layers[l].layerName, ext.extensionName));
			}
		}
	}
}}}
