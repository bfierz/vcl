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
#include <vcl/compute/opencl/commandqueue.h>

// VCL 
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace OpenCL
{
	CommandQueue::CommandQueue(Context* owner)
	: Compute::CommandQueue()
	, _ownerCtx(owner)
	{
		cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

		cl_int err;
		_queue = clCreateCommandQueue(*_ownerCtx, _ownerCtx->device(), properties, &err);
	}

	CommandQueue::~CommandQueue()
	{
		VCL_CL_SAFE_CALL(clReleaseCommandQueue(_queue));
	}

	void CommandQueue::sync()
	{

	}

	void CommandQueue::read(void* dst, Vcl::Compute::BufferView& src, size_t offset, size_t size, bool blocking)
	{

	}

	void CommandQueue::write(Vcl::Compute::BufferView& dst, void* src, size_t offset, size_t size, bool blocking)
	{

	}

	void CommandQueue::fill(Vcl::Compute::BufferView& dst, const void* pattern, size_t pattern_size)
	{

	}
}}}
