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
#include <array>
#include <string>
#include <vector>

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/graphics/vulkan/context.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{	
	class Memory final
	{
	public:
		//! Constructor
		Memory(Context* ctx, size_t size, int heapIndex, VkMemoryPropertyFlags flags);

		//! Destructor
		~Memory();

		//! Convert to Vulkan ID
		inline operator VkDeviceMemory() const
		{
			return _memory;
		}

		//! Access the pointer to the context object
		Context* context() const { return _context; }

	public:
		//! \returns the size in bytes
		size_t sizeInBytes() const { return _sizeInBytes; }

	public:
		void* map(size_t offset, size_t length);
		void unmap();
		
	private:
		//! Vulkan device
		Context* _context{ nullptr };

		//! Vulkan physical memory
		VkDeviceMemory _memory{ nullptr };

		//! Size of the memory region
		size_t _sizeInBytes{ 0 };

		//! Memory configuration
		VkMemoryPropertyFlags _memoryFlags{ 0 };
	};
}}}
