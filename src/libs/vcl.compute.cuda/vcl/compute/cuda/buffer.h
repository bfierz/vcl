/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#include <vcl/config/cuda.h>

// C++ standard library
#include <string>

// VCL
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/buffer.h>

namespace Vcl { namespace Compute { namespace Cuda {
	class Buffer : public Compute::Buffer
	{
	public:
		Buffer(Context* owner, BufferAccess hostAccess, size_t size);
		virtual ~Buffer();

	public:
		void resize(size_t new_size);

	public:
		//! Convert to OpenCL buffer ID
		inline operator CUdeviceptr() const
		{
			return _devicePtr;
		}

		CUdeviceptr devicePtr() const { return _devicePtr; }

	private:
		void allocate();
		void free();

	private:
		//! Link to the owning CL context
		Context* _ownerCtx{ nullptr };

		//! Cuda memory object
		CUdeviceptr _devicePtr{ 0 };
	};
}}}
