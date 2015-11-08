/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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

// VCL
#include <vcl/core/flags.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	enum class Usage
	{
		/*! 
		 * The resource is readable and writeable by the .
		 * This buffer is not mappabl
		 */
		Default = 0,
		
		/*! 
		 * The resource is not accessible by the CPU.
		 * And cannot be updated by the GPU (it can only be read by it). 
		 */
		Immutable = 1,

		/*!
		 * The resource can be mapped to CPU memory to be frequently written to
		 */
		Dynamic = 2,
		Staging = 3
	};
	
	VCL_DECLARE_FLAGS(CPUAccess, 
		
		/*! 
		 * The resource is to be mappable so that the CPU can change its contents.
		 */
		Write,

		/*!
		 * The resource is to be mappable so that the CPU can read its contents.
		 */
		Read
	);

	VCL_DECLARE_FLAGS(MapOptions,

		/*!
		 * Map the resource and invalidate the mapped range.
		 */
		InvalidateRange
	);

	struct BufferDescription
	{
		uint32_t SizeInBytes;
		Usage Usage;
		Flags<CPUAccess> CPUAccess;
	};

	struct BufferInitData
	{
		//! Pointer to a buffer containing the initial data
		void* Data;

		//! Size of the data buffer
		size_t SizeInBytes;
	};

	class Buffer
	{
	protected:
		Buffer(const BufferDescription& desc);

	public:
		virtual ~Buffer() = default;

	public:

		//! \returns the assigned usage of the buffer
		Usage usage() const { return _usage; }

		//! \returns the valid CPU access
		Flags<CPUAccess> cpuAccess() const { return _cpuAccess; }

		//! \returns the size in bytes
		int sizeInBytes() const { return _sizeInBytes; }

	private:
		//! Buffer usage
		Usage _usage;

		//! Allows CPU access pattern
		Flags<CPUAccess> _cpuAccess;

	protected:
		//! Size in bytes
		int _sizeInBytes{ 0 };
	};
}}}
