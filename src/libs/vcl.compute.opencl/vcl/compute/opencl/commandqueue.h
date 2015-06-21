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
#include <vcl/config/opencl.h>

// C++ standard library
#include <string>

// VCL
#include <vcl/compute/opencl/context.h>
#include <vcl/compute/commandqueue.h>

namespace Vcl { namespace Compute { namespace OpenCL
{	
	class CommandQueue : public Compute::CommandQueue
	{
	public:
		CommandQueue(Context* owner);
		virtual ~CommandQueue();

	public:
		//! Convert to OpenCL command queue ID
		inline operator cl_command_queue() const
		{
			return _queue;
		}

	public:
		virtual void sync() override;

	public:
		virtual void read(void* dst, Vcl::Compute::BufferView& src, size_t offset, size_t size, bool blocking = false) override;
		virtual void write(Vcl::Compute::BufferView& dst, void* src, size_t offset, size_t size, bool blocking = false) override;

		virtual void fill(Vcl::Compute::BufferView& dst, const void* pattern, size_t pattern_size) override;

	private:
		//! Link to the owning CL context
		Context* _ownerCtx{ nullptr };

		//! Native command queue handle
		cl_command_queue _queue{ nullptr };
	};
}}}
