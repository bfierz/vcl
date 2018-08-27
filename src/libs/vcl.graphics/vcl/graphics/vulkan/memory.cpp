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
#include <vcl/graphics/vulkan/memory.h>

// C++ standard library
#include <vector>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	MemoryHeap::MemoryHeap(const Device* dev, size_t size, int heap_index, int type_index, MemoryHeapType type, VkMemoryPropertyFlags flags)
	: _device(dev)
	, _type(type)
	, _sizeInBytes(size)
	, _vkMemoryFlags(flags)
	, _vkHeapIndex(heap_index)
	, _vkTypeIndex(type_index)
	{

	}

	Memory::Memory(Context* ctx, const MemoryHeap* heap, size_t size)
	: _context(ctx)
	, _sizeInBytes(size)
	, _heap(heap)
	{
		VkMemoryAllocateInfo alloc = {};
		alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc.allocationSize = _sizeInBytes;
		alloc.memoryTypeIndex = heap->heapTypeIndex();

		VkResult res = vkAllocateMemory(*_context, &alloc, nullptr, &_memory);
		VclEnsure(res == VK_SUCCESS, "Memory was allocated.");
	}

	Memory::~Memory()
	{
		VclRequire(_memory, "Memory is allocated.");

		if (_memory)
		{
			vkFreeMemory(*_context, _memory, nullptr);
			_memory = nullptr;
		}

		VclEnsure(!_memory, "Memory is cleaned up.");
	}

	void* Memory::map(size_t offset, size_t length)
	{
		VclRequire(offset + length <= sizeInBytes(), "Map request lies in range");

		if ((_heap->properties() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
		{
			throw "Buffer is not mappable.";
		}

		void* mapped_ptr = nullptr;
		vkMapMemory(*_context, _memory, offset, length, 0, &mapped_ptr);

		VclEnsure(mapped_ptr != nullptr, "Memory map is valid");
		return mapped_ptr;
	}

	void Memory::unmap()
	{
		vkUnmapMemory(*_context, _memory);
	}
}}}
