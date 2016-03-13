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
#include <vcl/graphics/vulkan/context.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	class Device final
	{
	public:
		//! Constructor
		Device(VkPhysicalDevice dev);

		// Remove move and copy constructors
		Device(const Device&) = delete;

		//! Default move ctor allows to create temporary objects
		Device(Device&&) = default;

		//! Destructor
		~Device() = default;

		// Remove move and copy assignment operators
		Device& operator=(const Device&) = delete;
		Device& operator=(Device&&) = delete;

		//! Convert to Vulkan ID
		inline operator VkPhysicalDevice() const
		{
			return _device;
		}

	public:
		//! \returns a new context object
		std::unique_ptr<Context> createContext(gsl::span<const char*> extensions = gsl::span<const char*>{});

	public:
		//! \returns the name of this device
		const std::string& name() const { return _name; }
		
		//! \returns the available memory types
		gsl::span<const VkMemoryType> memoryTypes() const;

	public:
		uint32_t getMemoryTypeIndex(uint32_t typeBits, VkFlags properties);

	private:
		static void enumerateLayersAndExtensions
		(
			VkPhysicalDevice device,
			std::vector<VkLayerProperties>& layers,
			std::vector<VkExtensionProperties>& availableExtensions,
			std::multimap<std::string, std::string>& extensionsPerLayer
		);

	private:
		//! Vulkan phyiscal device
		VkPhysicalDevice _device{ nullptr };

		//! Name of this device
		std::string _name;

		//! Available extensions
		std::vector<VkLayerProperties> _availableLayers;

		//! Available instance extensions
		std::vector<VkExtensionProperties> _availableExtensions;

		//! Extensions for each layer
		std::multimap<std::string, std::string> _extensionsPerLayer;

		//! Available queue families
		std::vector<VkQueueFamilyProperties> _queueFamilies;

		//! Device features
		VkPhysicalDeviceFeatures _features;

		//! Device memory configuration
		VkPhysicalDeviceMemoryProperties _memory;
	};
}}}
