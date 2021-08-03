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

namespace Vcl { namespace Graphics { namespace Runtime {
	VCL_DECLARE_FLAGS(BufferUsage,

					  //! Buffer is mappable so the it is readable by the CPU
					  MapRead,

					  //! Buffer is mappable so the it is writable by the CPU
					  MapWrite,

					  //! Buffer can be the source of copy opertions
					  CopySrc,

					  //! Buffer can be the destination of copy opertions
					  CopyDst,

					  //! Buffer can be used as index buffer
					  Index,

					  //! Buffer can be used as vertex buffer
					  Vertex,

					  //! Buffer can be used as uniform buffer
					  Uniform,

					  //! Buffer can be used as generic buffer in shaders
					  Storage,

					  //! Buffer can be used as source for indirect draw calls
					  Indirect,

					  //! Buffer can be used to stream data out
					  StreamOut)

	struct BufferDescription
	{
		uint32_t SizeInBytes;
		Flags<BufferUsage> Usage;
	};

	struct BufferInitData
	{
		//! Pointer to a buffer containing the initial data
		const void* Data;

		//! Size of the data buffer
		size_t SizeInBytes;
	};

	class Buffer
	{
	protected:
		Buffer(size_t size, Flags<BufferUsage> usage);

	public:
		virtual ~Buffer() = default;

	public:
		//! \returns the assigned usage of the buffer
		Flags<BufferUsage> usage() const { return _usage2; }

	public:
		//! \returns the size in bytes
		size_t sizeInBytes() const { return _sizeInBytes; }

	private:
		//! Size in bytes
		size_t _sizeInBytes{ 0 };

		//! Buffer usage
		Flags<BufferUsage> _usage2;
	};
}}}
