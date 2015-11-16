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
#include <vcl/config/opengl.h>

// C++ standard library
#include <array>

// VCL
#include <vcl/graphics/runtime/opengl/resource/resource.h>
#include <vcl/graphics/runtime/resource/buffer.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	VCL_DECLARE_FLAGS(MapOptions,
	
		/*!
		 * Map the resource unsynchronized into host memory. Mapped even when in use (no stall).
		 */
		Unsynchronized,
		
		/*!
		 * Map the resource persistently into host memory
		 */
		Persistent,

		/*!
		 * Coherent state will be reached explicitly after flushing or unmapping
		 */
		ExplicitFlush,
		
		/*!
		 * Operations are made visible to the device automatically
		 */
		CoherentWrite,
	
		/*!
		 * Map the resource and invalidate the entire buffer.
		 */
		InvalidateBuffer,

		/*!
		 * Map the resource and invalidate the mapped range.
		 */
		InvalidateRange
	);

	class Buffer : public Runtime::Buffer, public Resource
	{
	public:
		Buffer(const BufferDescription& desc, bool allowPersistentMapping = false, bool allowCoherentMapping = false, const BufferInitData* init_data = nullptr);
		virtual ~Buffer();

	public:
		void* map(size_t offset, size_t length, Flags<CPUAccess> access = CPUAccess::Write, Flags<MapOptions> options = {});
		void unmap();

		/*!
		 * \param offset Offset to the beginning of the currently mapped range
		 */
		void flushRange(size_t offset = 0, size_t length = std::numeric_limits<size_t>::max());

	public:
		void copyTo(Buffer& target, size_t srcOffset = 0, size_t dstOffset = 0, size_t size = std::numeric_limits<size_t>::max());

	private:
		//! Buffer can be mapped persistently
		bool _allowPersistentMapping{ false };

		//! Buffer can be mapped coherently
		bool _allowCoherentMapping{ false };

	private:
		//! Flags indicating with which host access the buffer was mapped
		Flags<CPUAccess> _mappedAccess;

		//! Flags indicating with which host access the buffer was mapped
		Flags<MapOptions> _mappedOptions;

		//! Offset of the mapped memory region
		size_t _mappedOffset;

		//! Size of the mapped memory region
		size_t _mappedSize;
	};
}}}}
#endif // VCL_OPENGL_SUPPORT
