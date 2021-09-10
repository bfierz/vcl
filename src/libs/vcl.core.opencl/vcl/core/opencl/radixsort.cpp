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
#include <vcl/core/opencl/radixsort.h>

// VCL
#include <vcl/compute/opencl/buffer.h>
#include <vcl/compute/opencl/kernel.h>
#include <vcl/compute/opencl/module.h>

extern uint32_t RadixSortCL[];
extern size_t RadixSortCLSize;

namespace Vcl { namespace Core { namespace OpenCL {
	RadixSort::RadixSort(Vcl::Compute::OpenCL::Context* ctx, unsigned int maxElements)
	: _ownerCtx(ctx)
	, _scan(ctx, maxElements / 2 / LocalSize * 16)
	{
		unsigned int numBlocks = ((maxElements % (LocalSize * 4)) == 0) ? (maxElements / (LocalSize * 4)) : (maxElements / (LocalSize * 4) + 1);

		_tmpKeys = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, sizeof(unsigned int) * maxElements);
		_counters = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, WarpSize * numBlocks * sizeof(unsigned int));
		_countersSum = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, WarpSize * numBlocks * sizeof(unsigned int));
		_blockOffsets = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, WarpSize * numBlocks * sizeof(unsigned int));

		// Load the module
		_radixSortModule = ctx->createModuleFromSource(reinterpret_cast<const int8_t*>(RadixSortCL), RadixSortCLSize * sizeof(uint32_t));

		if (_radixSortModule)
		{
			// Load the sorting kernels
			_radixSortBlocksKeysOnlyKernel = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Kernel>(_radixSortModule->kernel("radixSortBlocksKeysOnly"));
			_findRadixOffsetsKernel = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Kernel>(_radixSortModule->kernel("findRadixOffsets"));
			_scanNaiveKernel = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Kernel>(_radixSortModule->kernel("scanNaive"));
			_reorderDataKeysOnlyKernel = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Kernel>(_radixSortModule->kernel("reorderDataKeysOnly"));
		}
	}

	void RadixSort::operator()(
		ref_ptr<Compute::Buffer> keys,
		unsigned int numElements,
		unsigned int keyBits)
	{
		radixSortKeysOnly(keys, numElements, keyBits);
	}

	void RadixSort::radixSortKeysOnly(ref_ptr<Compute::Buffer> keys, unsigned int numElements, unsigned int keyBits)
	{
		const unsigned int bitStep = 4;

		int i = 0;
		while (keyBits > i * bitStep)
		{
			radixSortStepKeysOnly(keys, bitStep, i * bitStep, numElements);
			i++;
		}
	}

	void RadixSort::radixSortStepKeysOnly(ref_ptr<Compute::Buffer> keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
	{
		// Four step algorithms from Satish, Harris & Garland
		radixSortBlocksKeysOnlyOCL(keys, nbits, startbit, numElements);

		findRadixOffsetsOCL(startbit, numElements);

		_scan(_countersSum, _counters, 1, numElements / 2 / LocalSize * 16);

		reorderDataKeysOnlyOCL(keys, startbit, numElements);
	}

	void RadixSort::radixSortBlocksKeysOnlyOCL(ref_ptr<Compute::Buffer> keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
	{
		VclRequire(_radixSortBlocksKeysOnlyKernel, "Kernel is loaded.");
		using Vcl::Compute::OpenCL::LocalMemory;

		auto bufKeys = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(keys);
		auto bufTmpKeys = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_tmpKeys);

		unsigned int totalBlocks = numElements / 4 / LocalSize;
		std::array<size_t, 3> globalWorkSize = { LocalSize * totalBlocks, 0, 0 };
		std::array<size_t, 3> localWorkSize = { LocalSize, 0, 0 };

		_radixSortBlocksKeysOnlyKernel->run(
			*Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::CommandQueue>(_ownerCtx->defaultQueue()),
			1,
			globalWorkSize,
			localWorkSize,
			(cl_mem)*bufKeys,
			(cl_mem)*bufTmpKeys,
			nbits,
			startbit,
			numElements,
			totalBlocks,
			LocalMemory(4 * LocalSize * sizeof(unsigned int)));
	}

	void RadixSort::findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements)
	{
		VclRequire(_findRadixOffsetsKernel, "Kernel is loaded.");
		using Vcl::Compute::OpenCL::LocalMemory;

		auto bufTmpKeys = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_tmpKeys);
		auto bufCounters = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_counters);
		auto bufBlockOffsets = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_blockOffsets);

		unsigned int totalBlocks = numElements / 2 / LocalSize;
		std::array<size_t, 3> globalWorkSize = { LocalSize * totalBlocks, 0, 0 };
		std::array<size_t, 3> localWorkSize = { LocalSize, 0, 0 };

		_findRadixOffsetsKernel->run(
			*Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::CommandQueue>(_ownerCtx->defaultQueue()),
			1,
			globalWorkSize,
			localWorkSize,
			(cl_mem)*bufTmpKeys,
			(cl_mem)*bufCounters,
			(cl_mem)*bufBlockOffsets,
			startbit,
			numElements,
			totalBlocks,
			LocalMemory(2 * LocalSize * sizeof(unsigned int)));
	}

#define NUM_BANKS 16
	void RadixSort::scanNaiveOCL(unsigned int numElements)
	{
		VclRequire(_scanNaiveKernel, "Kernel is loaded.");
		using Vcl::Compute::OpenCL::LocalMemory;

		auto bufCounters = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_counters);
		auto bufCountersSum = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_countersSum);

		unsigned int nHist = numElements / 2 / LocalSize * 16;
		std::array<size_t, 3> globalWorkSize = { nHist, 0, 0 };
		std::array<size_t, 3> localWorkSize = { nHist, 0, 0 };
		unsigned int extra_space = nHist / NUM_BANKS;
		unsigned int shared_mem_size = sizeof(unsigned int) * (nHist + extra_space);

		_scanNaiveKernel->run(
			*Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::CommandQueue>(_ownerCtx->defaultQueue()),
			1,
			globalWorkSize,
			localWorkSize,
			(cl_mem)*bufCountersSum,
			(cl_mem)*bufCounters,
			nHist,
			LocalMemory(2 * shared_mem_size));
	}

	void RadixSort::reorderDataKeysOnlyOCL(ref_ptr<Compute::Buffer> keys, unsigned int startbit, unsigned int numElements)
	{
		VclRequire(_reorderDataKeysOnlyKernel, "Kernel is loaded.");
		using Vcl::Compute::OpenCL::LocalMemory;

		auto bufKeys = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(keys);
		auto bufTmpKeys = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_tmpKeys);
		auto bufBlockOffsets = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_blockOffsets);
		auto bufCounters = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_counters);
		auto bufCountersSum = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_countersSum);

		unsigned int totalBlocks = numElements / 2 / LocalSize;
		std::array<size_t, 3> globalWorkSize = { LocalSize * totalBlocks, 0, 0 };
		std::array<size_t, 3> localWorkSize = { LocalSize, 0, 0 };

		_reorderDataKeysOnlyKernel->run(
			*Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::CommandQueue>(_ownerCtx->defaultQueue()),
			1,
			globalWorkSize,
			localWorkSize,
			(cl_mem)*bufKeys,
			(cl_mem)*bufTmpKeys,
			(cl_mem)*bufBlockOffsets,
			(cl_mem)*bufCountersSum,
			(cl_mem)*bufCounters,
			startbit,
			numElements,
			totalBlocks,
			LocalMemory(2 * LocalSize * sizeof(unsigned int)));
	}
}}}
