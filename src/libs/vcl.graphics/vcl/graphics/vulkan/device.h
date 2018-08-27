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
#include <vcl/core/span.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/memory.h>

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


	public:
		//! \returns the name of this device
		const std::string& name() const { return _name; }

		//! \returns a new context object
		std::unique_ptr<Context> createContext(stdext::span<const char*> extensions = stdext::span<const char*>{});

		//! Access the available memory heaps
		//! \returns A list of available memory heaps
		stdext::span<const MemoryHeap> memoryHeaps() const { return _memoryHeaps; }

		//! Find a memory heap that fulfills certain requirements
		//! \returns A compatible memory heap
		const MemoryHeap* findMemoryHeap(const VkMemoryRequirements& requirements, MemoryHeapType heap_type) const;

		/*!
		 * \name Native Vulkan API
		 * Access the underlying Vulkan data types
		 */
		//! \{
		//! Convert to Vulkan ID
		operator VkPhysicalDevice() const
		{
			return _device;
		}

		//! \returns the available memory types
		stdext::span<const VkMemoryType> nativeMemoryTypes() const;

		//! \returns the available memory heaps
		stdext::span<const VkMemoryHeap> nativeMemoryHeaps() const;

		//! \}

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

		//! Native device memory configuration
		VkPhysicalDeviceMemoryProperties _vkMemory;

		//! Sorted memory configurations
		std::vector<MemoryHeap> _memoryHeaps;
	};
}}}
