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
#include <vcl/compute/opencl/buffer.h>
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
		VCL_CL_SAFE_CALL(clFinish(_queue));
	}

	void CommandQueue::read(void* dst, Vcl::Compute::BufferView& src, bool blocking)
	{
		auto& buffer = src.owner();
		auto& clBuffer = static_cast<const Buffer&>(buffer);

		VCL_CL_SAFE_CALL(clEnqueueReadBuffer(_queue, (cl_mem) clBuffer, blocking, src.offset(), src.size(), dst, 0, nullptr, nullptr));
	}

	void CommandQueue::write(Vcl::Compute::BufferView& dst, void* src, bool blocking)
	{
		auto& buffer = dst.owner();
		auto& clBuffer = static_cast<const Buffer&>(buffer);

		VCL_CL_SAFE_CALL(clEnqueueWriteBuffer(_queue, (cl_mem) clBuffer, blocking, dst.offset(), dst.size(), src, 0, nullptr, nullptr));
	}

	void CommandQueue::fill(Vcl::Compute::BufferView& dst, const void* pattern, size_t pattern_size)
	{
		Require(dynamic_cast<const Buffer*>(&dst.owner()), "Buffer is OpenCL buffer.");
		Require(pattern_size == 1 || pattern_size == 2 || pattern_size == 4, "Valid pattern size.");

		auto& buffer = dst.owner();
		auto& clBuffer = static_cast<const Buffer&>(buffer);

		VCL_CL_SAFE_CALL(clEnqueueFillBuffer(_queue, (cl_mem) clBuffer, pattern, pattern_size, dst.offset(), dst.size(), 0, nullptr, nullptr));
	}
}}}
