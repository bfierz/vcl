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
#include <vcl/core/opencl/scan.h>

// VCL
#include <vcl/compute/opencl/buffer.h>
#include <vcl/compute/opencl/kernel.h>
#include <vcl/compute/opencl/module.h>

extern uint32_t ScanCL[];
extern size_t ScanCLSize;

namespace Vcl { namespace Core { namespace OpenCL
{
	namespace
	{
		unsigned int iSnapUp(unsigned int dividend, unsigned int divisor)
		{
			return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
		}

		unsigned int factorRadix2(unsigned int& log2L, unsigned int L)
		{
			if (!L)
			{
				log2L = 0;
				return 0;
			}
			else {
				for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
				return L;
			}
		}
	}

	ScanExclusiveLarge::ScanExclusiveLarge(Vcl::Compute::OpenCL::Context* ctx, unsigned int maxElements)
	: _ownerCtx(ctx)
	, _maxElements(maxElements)
	{
		const unsigned int WarpSize = 32;

		_workSpace = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, std::max(1u, maxElements / MaxWorkgroupInclusiveScanSize) * sizeof(unsigned int));

		// Load the module
		_scanModule = ctx->createModuleFromSource(reinterpret_cast<const int8_t*>(ScanCL), ScanCLSize * sizeof(uint32_t));

		if (_scanModule)
		{
			// Load the sorting kernels
			_scanExclusiveLocal1Kernel = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Kernel>(_scanModule->kernel("scanExclusiveLocal1"));
			_scanExclusiveLocal2Kernel = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Kernel>(_scanModule->kernel("scanExclusiveLocal2"));
			_uniformUpdateKernel = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Kernel>(_scanModule->kernel("uniformUpdate"));
		}
	}

	void ScanExclusiveLarge::operator()
	(
		ref_ptr<Compute::Buffer> dst,
		ref_ptr<Compute::Buffer> src,
		unsigned int batchSize,
		unsigned int arrayLength
	)
	{
		// Check power-of-two factorization
		unsigned int log2L;
		unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
		Check(factorizationRemainder == 1, "Is power of two");

		// Check supported size range
		Check((arrayLength >= MinLargeArraySize) && (arrayLength <= MaxLargeArraySize), "Array is within size");

		// Check total batch size limit
		Check((batchSize * arrayLength) <= MaxBatchElements, "Batch size is within range");

		scanExclusiveLocal1
		(
			dst,
			src,
			(batchSize * arrayLength) / (4 * WorkgroupSize),
			4 * WorkgroupSize
		);

		scanExclusiveLocal2
		(
			_workSpace,
			dst,
			src,
			batchSize,
			arrayLength / (4 * WorkgroupSize)
		);

		uniformUpdate
		(
			dst,
			_workSpace,
			(batchSize * arrayLength) / (4 * WorkgroupSize)
		);
	}
	
	void ScanExclusiveLarge::scanExclusiveLocal1
	(
		ref_ptr<Compute::Buffer> dst,
		ref_ptr<Compute::Buffer> src,
		unsigned int n,
		unsigned int size
	)
	{
		Require(_scanExclusiveLocal1Kernel, "Kernel is loaded.");
		using Vcl::Compute::OpenCL::LocalMemory;

		auto bufDst = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(dst);
		auto bufSrc = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(src);

		std::array<size_t, 3> globalWorkSize = { (n*size) / 4, 0, 0 };
		std::array<size_t, 3> localWorkSize = { WorkgroupSize, 0, 0 };

		_scanExclusiveLocal1Kernel->run
		(
			*Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::CommandQueue>(_ownerCtx->defaultQueue()),
			1,
			globalWorkSize,
			localWorkSize,
			(cl_mem) *bufDst,
			(cl_mem) *bufSrc,
			LocalMemory(2 * WorkgroupSize * sizeof(unsigned int)),
			size
		);
	}

	void ScanExclusiveLarge::scanExclusiveLocal2
	(
		ref_ptr<Compute::Buffer> buffer,
		ref_ptr<Compute::Buffer> dst,
		ref_ptr<Compute::Buffer> src,
		unsigned int n,
		unsigned int size
	)
	{
		Require(_scanExclusiveLocal2Kernel, "Kernel is loaded.");
		using Vcl::Compute::OpenCL::LocalMemory;

		auto bufBuf = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_workSpace);
		auto bufDst = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(dst);
		auto bufSrc = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(src);

		unsigned int elements = n * size;
		std::array<size_t, 3> globalWorkSize = { iSnapUp(elements, WorkgroupSize), 0, 0 };
		std::array<size_t, 3> localWorkSize = { WorkgroupSize, 0, 0 };

		_scanExclusiveLocal2Kernel->run
		(
			*Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::CommandQueue>(_ownerCtx->defaultQueue()),
			1,
			globalWorkSize,
			localWorkSize,
			(cl_mem) *bufBuf,
			(cl_mem) *bufDst,
			(cl_mem) *bufSrc,
			LocalMemory(2 * WorkgroupSize * sizeof(unsigned int)),
			elements,
			size
		);
	}

	void ScanExclusiveLarge::uniformUpdate
	(
		ref_ptr<Compute::Buffer> dst,
		ref_ptr<Compute::Buffer> buffer,
		unsigned int n
	)
	{
		Require(_uniformUpdateKernel, "Kernel is loaded.");
		using Vcl::Compute::OpenCL::LocalMemory;

		auto bufDst = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(dst);
		auto bufBuf = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_workSpace);

		std::array<size_t, 3> globalWorkSize = { n * WorkgroupSize, 0, 0 };
		std::array<size_t, 3> localWorkSize = { WorkgroupSize, 0, 0 };

		_uniformUpdateKernel->run
		(
			*Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::CommandQueue>(_ownerCtx->defaultQueue()),
			1,
			globalWorkSize,
			localWorkSize,
			(cl_mem) *bufDst,
			(cl_mem) *bufBuf
		);
	}
}}}
