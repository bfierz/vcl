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
#include <vcl/compute/cuda/commandqueue.h>

// VCL 
#include <vcl/compute/cuda/buffer.h>
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace Cuda
{
	CommandQueue::CommandQueue(Context* owner)
	: Compute::CommandQueue()
	, _ownerCtx(owner)
	{
		VCL_CU_SAFE_CALL(cuStreamCreate(&_queue, 0));
	}

	CommandQueue::~CommandQueue()
	{
		VCL_CU_SAFE_CALL(cuStreamDestroy(_queue));
	}

	void CommandQueue::sync()
	{
		VCL_CU_SAFE_CALL(cuStreamSynchronize(_queue));
	}

	void CommandQueue::copy(BufferView dst, ConstBufferView src)
	{
		VclRequire(dynamic_cast<const Buffer*>(&src.owner()), "src is CUDA buffer.");
		VclRequire(dynamic_cast<const Buffer*>(&dst.owner()), "dst is CUDA buffer.");
		VclRequire(src.offset() % 4 == 0, "src offset is aligned.");
		VclRequire(dst.offset() % 4 == 0, "dst ffset is aligned.");
		VclRequire(dst.size() >= src.size(), "Sizes of views match");

		auto& dstBuffer = static_cast<Buffer&>(dst.owner());
		auto& srcBuffer = static_cast<const Buffer&>(src.owner());

		VCL_CU_SAFE_CALL(cuMemcpyDtoDAsync((CUdeviceptr) dstBuffer + src.offset(), (CUdeviceptr) srcBuffer + src.offset(), src.size(), _queue));
	}

	void CommandQueue::read(void* dst, ConstBufferView src, bool blocking)
	{
		VclRequire(dynamic_cast<const Buffer*>(&src.owner()), "src is CUDA buffer.");
		VclRequire(src.offset() % 4 == 0, "Offset is aligned.");

		auto& cuBuffer = static_cast<const Buffer&>(src.owner());

		VCL_CU_SAFE_CALL(cuMemcpyDtoHAsync(dst, (CUdeviceptr) cuBuffer + src.offset(), src.size(), _queue));

		if (blocking)
			sync();
	}

	void CommandQueue::write(BufferView dst, const void* src, bool blocking)
	{
		VclRequire(dynamic_cast<const Buffer*>(&dst.owner()), "dst is CUDA buffer.");
		VclRequire(dst.offset() % 4 == 0, "Offset is aligned.");

		auto& cuBuffer = static_cast<Buffer&>(dst.owner());

		VCL_CU_SAFE_CALL(cuMemcpyHtoDAsync((CUdeviceptr) cuBuffer + dst.offset(), src, dst.size(), _queue));

		if (blocking)
			sync();
	}

	void CommandQueue::fill(BufferView dst, const void* pattern, size_t pattern_size)
	{
		VclRequire(dynamic_cast<const Buffer*>(&dst.owner()), "dst is CUDA buffer.");
		VclRequire(pattern_size == 1 || pattern_size == 2 || pattern_size == 4, "Valid pattern size.");

		auto& cuBuffer = static_cast<Buffer&>(dst.owner());

		if (pattern_size == 1)
		{
			VCL_CU_SAFE_CALL(cuMemsetD8Async((CUdeviceptr) cuBuffer + dst.offset(), *(const uint8_t*) pattern, dst.size() / sizeof(uint8_t), _queue));
		}
		else if (pattern_size == 2)
		{
			VCL_CU_SAFE_CALL(cuMemsetD16Async((CUdeviceptr) cuBuffer + dst.offset(), *(const uint16_t*) pattern, dst.size() / sizeof(uint16_t), _queue));
		}
		else if (pattern_size == 4)
		{
			VCL_CU_SAFE_CALL(cuMemsetD32Async((CUdeviceptr) cuBuffer + dst.offset(), *(const uint32_t*) pattern, dst.size() / sizeof(uint32_t), _queue));
		}
	}
}}}
