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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <map>
#include <memory>
#include <string>
#include <vector>

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/graphics/vulkan/device.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	struct PlatformDesc
	{
		const char* ApplicationName;

		uint32_t NrLayers;
		const char** Layers;

		uint32_t NrExtensions;
		const char** Extensions;
	};

	class Platform final
	{
	public:
		Platform(gsl::span<const char*> extensions = {});
		~Platform();

		//! Convert to Vulkan ID
		inline operator VkInstance() const
		{
			return _instance;
		}

	public: // Query layers
		const std::vector<VkLayerProperties>& availableLayers() { return _availableLayers; }
		const std::vector<VkExtensionProperties>& availableExtensions() { return _availableExtensions; }

	public:
		int nrDevices() const;
		Device& device(int idx);

	private:
		//! Vulkan instance pointer
		VkInstance _instance{ nullptr };

		//! List of available vulkan devices
		std::vector<Device> _devices;

		//! Debug callback of this platform instance
		VkDebugReportCallbackEXT _debugCallback;

	private: // Debug extension callbacks
		PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT;
		PFN_vkDebugReportMessageEXT vkDebugReportMessageEXT;
		PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT;

	private:
		//! Available extensions
		std::vector<VkLayerProperties> _availableLayers;

		//! Available instance extensions
		std::vector<VkExtensionProperties> _availableExtensions;

		//! Extensions for each layer
		std::multimap<std::string, std::string> _extensionsPerLayer;
	};
}}}
