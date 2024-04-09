/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/vulkan/memory.h>
#include <vcl/graphics/vulkan/platform.h>

// Map the memory heap types to strings
const std::array<std::string, 6> MemoryHeapTypeString =
{
	"None",
	"Default",
	"Streaming",
	"Upload",
	"Download",
	"Other"
};

// Enumerate the available memory types
int main(int argc, char* argv[])
{
	using namespace Vcl::Graphics::Vulkan;

	// Vulkan extension
	std::vector<const char*> platform_extensions = { };
	std::vector<const char*> context_extensions = { };

	// Initialize the Vulkan platform
	auto platform = std::make_unique<Platform>(stdext::make_span(platform_extensions));
	for (auto& device : platform->devices())
	{
		std::cout << "Device: " << device.name() << "\n";

		// Print the available memory strucure
		const auto memory_heaps = device.nativeMemoryHeaps();
		const auto memory_types = device.memoryHeaps();

		int heap_index = 0;
		for (const auto& heap : memory_heaps)
		{
			std::cout << "Memory heap " << heap_index << ":\n";
			std::cout << "* Heap size:" << heap.size << "\n";
			std::cout << "* Heap memory type:";
			if ((heap.flags & VkMemoryHeapFlagBits::VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
			{
				std::cout << "device memory\n";
			}
			else
			{
				std::cout << "host memory\n";
			}

			// Iterate through the available memory types
			std::cout << "* Available memory types:\n";
			for (const auto& type : memory_types)
			{
				if (type.heapIndex() != heap_index)
				{
					continue;
				}
				VclCheck(((type.properties() & VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) ==
					     ((heap.flags & VkMemoryHeapFlagBits::VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0),
					     "Memory location is consistent.");

				const bool visible =  (type.properties() & VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
				const bool coherent = (type.properties() & VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
				const bool cached =   (type.properties() & VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
				const bool lazy =     (type.properties() & VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0;

				std::cout << "  * Memory type: " << MemoryHeapTypeString[int(type.type())] << "\n";
				std::cout << "  * Memory type index:" << type.heapTypeIndex() << "\n";
				std::cout << "  * Memory is host visible:" << visible << "\n";
				std::cout << "  * Memory is cached:" << cached << "\n";
				std::cout << "  * Memory is coherent:" << coherent << "\n";
				std::cout << "  * Memory is lazily allocated:" << lazy << "\n\n";
			}
			heap_index++;
		}
	}

	return 0;
}
