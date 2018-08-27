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

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/core/span.h>
#include <vcl/graphics/vulkan/context.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	enum class MemoryHeapType
	{
		//! Inaccessible memory heap type
		None,

		//! Fast device accessible memory, might provide direct host access
		Default,

		//! Memory used to stream data from the host to the device
		Streaming,

		//! Memory used to upload to the device, optimized for memory writing
		Upload,

		//! Memory used to download data from the device, optizimed for reading
		Download,

		//! Any other type of memory, needs further querying
		Other
	};

	class MemoryHeap final
	{
	public:
		//! Constructor
		MemoryHeap(const Device* dev, size_t size, int heap_index, int type_index, MemoryHeapType type, VkMemoryPropertyFlags flags);

	public:
		//! Semantic type of the heap
		//! \returns The semantic type of the heap
		MemoryHeapType type() const { return _type; }

		//! Index of the heap
		//! \returns The index of the heap
		uint32_t heapIndex() const { return _vkHeapIndex; }

		//! Index of the heap type
		//! \returns The index of the heap type
		uint32_t heapTypeIndex() const { return _vkTypeIndex; }

		//! Proprties of the heap
		//! \returns The properties of the heap
		VkMemoryPropertyFlags properties() const { return _vkMemoryFlags; }

	private:
		//! Owning Vulkan device
		const Device* _device{ nullptr };

		//! Heap semantics
		MemoryHeapType _type{ MemoryHeapType::None };

		//! Maximum heap size
		size_t _sizeInBytes{ 0 };

		//! Vulkan memory heap flags
		VkMemoryPropertyFlags _vkMemoryFlags{ 0 };

		//! Index of the heap
		uint32_t _vkHeapIndex{ 0xffffffff };

		//! Index of the heap type
		uint32_t _vkTypeIndex{ 0xffffffff };
	};

	class Memory final
	{
	public:
		//! Constructor
		Memory(Context* ctx, const MemoryHeap* heap, size_t size);

		//! Destructor
		~Memory();

		//! Convert to Vulkan ID
		operator VkDeviceMemory() const { return _memory; }

		//! Access the pointer to the context object
		//! \returns The context object
		Context* context() const { return _context; }

		//! Access the associated heap
		//! \returns The heap on which the memory was allocated
		const MemoryHeap* heapIndex() const { return _heap; }

		//! Access the size of the allocated memory
		//! \returns The size in bytes
		size_t sizeInBytes() const { return _sizeInBytes; }

	public:
		void* map(size_t offset, size_t length);
		void unmap();
		
	private:
		//! Owning Vulkan context
		Context* _context{ nullptr };

		//! Heap on which the memory is allocated
		const MemoryHeap* _heap{ nullptr };

		//! Vulkan physical memory
		VkDeviceMemory _memory{ nullptr };

		//! Size of the memory region
		size_t _sizeInBytes{ 0 };
	};
}}}
