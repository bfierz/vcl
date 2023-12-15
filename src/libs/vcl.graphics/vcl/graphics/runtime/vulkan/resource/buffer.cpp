/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2016 Basil Fierz
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
#include <vcl/graphics/runtime/vulkan/resource/buffer.h>

// C++ standard library

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/memory.h>

#ifdef VCL_VULKAN_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan
{
	VkBufferUsageFlags convert(Flags<BufferUsage> usage)
	{
		VkBufferUsageFlags flags = 0;

		if (usage.isSet(BufferUsage::CopySrc))      flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		if (usage.isSet(BufferUsage::CopyDst))      flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		if (usage.isSet(BufferUsage::UniformTexel)) flags |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
		if (usage.isSet(BufferUsage::StorageTexel)) flags |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
		if (usage.isSet(BufferUsage::Uniform))      flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		if (usage.isSet(BufferUsage::Storage))      flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		if (usage.isSet(BufferUsage::Index))        flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
		if (usage.isSet(BufferUsage::Vertex))       flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		if (usage.isSet(BufferUsage::Indirect))     flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

		return flags;
	}

	//! Determine the heap-type according to the buffer usage.
	//! Mappable buffers are mutually exclusive and have dedicated heap-types
	Vcl::Graphics::Vulkan::MemoryHeapType determineHeapType(Flags<BufferUsage> usage)
	{
		using Vcl::Graphics::Vulkan::MemoryHeapType;

		if (usage.isSet(BufferUsage::MapRead))
			return MemoryHeapType::Download;
		else if (usage.isSet(BufferUsage::MapWrite))
			return MemoryHeapType::Upload;
		else
			return MemoryHeapType::Default;
	}

	Buffer::Buffer
	(
		Vcl::Graphics::Vulkan::Context* context,
		const BufferDescription& desc,
		const BufferInitData* init_data,
		Vcl::Graphics::Vulkan::Memory* memory
	)
	: Runtime::Buffer(desc.SizeInBytes, desc.Usage)
	{
		VclRequire(implies(init_data, usage().isSet(BufferUsage::MapWrite)), "Initializaiton data requires write mappable buffer.");
		VclRequire(implies(init_data, init_data->SizeInBytes == desc.SizeInBytes), "Initialization data has same size as buffer.");
		
		VkBufferCreateInfo buffer_info;
		buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_info.pNext = nullptr;
		buffer_info.flags = 0;
		buffer_info.size = desc.SizeInBytes;
		buffer_info.usage = convert(usage());
		buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Buffer not shared between queues
		buffer_info.queueFamilyIndexCount = 0;
		buffer_info.pQueueFamilyIndices = nullptr;

		VkResult res;
		res = vkCreateBuffer(*context, &buffer_info, nullptr, &_buffer);
		VclCheck(res == VK_SUCCESS, "Buffer was created.");

		if (!memory)
		{
			VkMemoryRequirements reqs = {};
			vkGetBufferMemoryRequirements(*context, _buffer, &reqs);

			// Determine the vulkan memory flags
			using Vcl::Graphics::Vulkan::MemoryHeapType;
			using Vcl::Graphics::Vulkan::MemoryHeap;

			MemoryHeapType heap_type = determineHeapType(usage());
			const MemoryHeap* heap = context->device()->findMemoryHeap(reqs, heap_type);
			if (!heap && (heap_type == MemoryHeapType::Upload || heap_type == MemoryHeapType::Download))
			{
				heap = context->device()->findMemoryHeap(reqs, MemoryHeapType::Streaming);
			}

			VclCheck(heap, "Suitable heap found.");
			_memoryOwner = std::make_unique<Vcl::Graphics::Vulkan::Memory>(context, heap, desc.SizeInBytes);
			_memory = _memoryOwner.get();
		}
		else
		{
			_memory = memory;
		}

		if (init_data)
		{
			// Check if memory is host-accessible. If not, a temporary memory region
			// has to be used.
			//if (context->device()->memoryTypes()[_memory->heapIndex()].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT == 0)
			//{
			//	const auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			//	const int heap_index = context->device()->getMemoryTypeIndex(0, flags);
			//	Vcl::Graphics::Vulkan::Memory temp(context, init_data->SizeInBytes, _memory->heapIndex(), flags);
			//
			//	void* data = temp.map(0, init_data->SizeInBytes);
			//	memcpy(data, init_data->Data, (size_t)init_data->SizeInBytes);
			//	temp.unmap();
			//}			

			void* data = _memory->map(0, init_data->SizeInBytes);
			memcpy(data, init_data->Data, (size_t)init_data->SizeInBytes);
			_memory->unmap();
		}

		res = vkBindBufferMemory(*context, _buffer, *_memory, 0);
		VclCheck(res == VK_SUCCESS, "Buffer was bound to memory.");
	}

	Buffer::~Buffer()
	{
		VclRequire(_buffer, "Buffer is created.");
		VclRequire(_memory, "Memory is allocated.");
		
		if (_buffer)
		{
			vkDestroyBuffer(*(_memory->context()), _buffer, nullptr);
			_buffer = nullptr;
		}

		_memoryOwner.reset();
		_memory = nullptr;

		VclEnsure(!_memoryOwner, "Memory is cleaned up.");
		VclEnsure(!_buffer, "Buffer is cleaned up.");
	}
}}}}
#endif // VCL_VULKAN_SUPPORT
