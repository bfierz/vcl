/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#include <vcl/graphics/opengl/algorithm/radixsort.h>

// VCL
#include <vcl/core/contract.h>

// Compute shader
#include "scan.glslinc"
#include "radixsort.comp"

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics
{
	RadixSort::RadixSort(unsigned int maxElements)
	: _scan(maxElements * 16 / 2 / LocalSize)
	{
		using namespace Vcl::Graphics::Runtime;

		unsigned int numBlocks = ((maxElements % (LocalSize * 4)) == 0) ?
			(maxElements / (LocalSize * 4)) : (maxElements / (LocalSize * 4) + 1);

		BufferDescription desc_large =
		{
			maxElements * sizeof(unsigned int),
			ResourceUsage::Default,
			{}
		};
		BufferDescription desc_small =
		{
			32 * numBlocks * sizeof(unsigned int),
			ResourceUsage::Default,
			{}
		};
		
		_tmpKeys      = make_owner<OpenGL::Buffer>(desc_large);
		_counters     = make_owner<OpenGL::Buffer>(desc_small);
		_countersSum  = make_owner<OpenGL::Buffer>(desc_small);
		_blockOffsets = make_owner<OpenGL::Buffer>(desc_small);

		// Load the sorting kernels
		_radixSortBlocksKeysOnlyKernel = OpenGL::createComputeKernel(module, { "#define WORKGROUP_SIZE 128\n#define SCAN_SHARED_MEM_SIZE 4*WORKGROUP_SIZE\n#define radixSortBlocksKeysOnly\n", module_scan });
		_findRadixOffsetsKernel        = OpenGL::createComputeKernel(module, { "#define WORKGROUP_SIZE 128\n#define SCAN_SHARED_MEM_SIZE 4*WORKGROUP_SIZE\n#define findRadixOffsets\n", module_scan });
		_reorderDataKeysOnlyKernel     = OpenGL::createComputeKernel(module, { "#define WORKGROUP_SIZE 128\n#define SCAN_SHARED_MEM_SIZE 4*WORKGROUP_SIZE\n#define reorderDataKeysOnly\n", module_scan });
	}

	void RadixSort::operator()
	(
		ref_ptr<Runtime::OpenGL::Buffer> keys,
		unsigned int numElements,
		unsigned int keyBits
	)
	{
		radixSortKeysOnly(keys, numElements, keyBits);
	}

	void RadixSort::radixSortKeysOnly(ref_ptr<Runtime::OpenGL::Buffer> keys, unsigned int numElements, unsigned int keyBits)
	{
		const unsigned int bitStep = 4;

		int i = 0;
		while (keyBits > i*bitStep)
		{
			radixSortStepKeysOnly(keys, bitStep, i*bitStep, numElements);
			i++;
		}
	}

	void RadixSort::radixSortStepKeysOnly(ref_ptr<Runtime::OpenGL::Buffer> keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
	{
		// Four step algorithms from Satish, Harris & Garland
		radixSortBlocksKeysOnlyOCL(keys, nbits, startbit, numElements);

		findRadixOffsetsOCL(startbit, numElements);

		unsigned int array_length = numElements * 16 / 2 / LocalSize;
		_scan(_countersSum, _counters, array_length);
		
		reorderDataKeysOnlyOCL(keys, startbit, numElements);
	}

	void RadixSort::radixSortBlocksKeysOnlyOCL(ref_ptr<Runtime::OpenGL::Buffer> keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
	{
		Require(_radixSortBlocksKeysOnlyKernel, "Kernel is loaded.");

		unsigned int totalBlocks = std::max<unsigned int>(numElements / 4 / LocalSize, 1);

		// Bind the program to the pipeline
		_radixSortBlocksKeysOnlyKernel->bind();

		// Bind the buffers and parameters
		_radixSortBlocksKeysOnlyKernel->setBuffer("KeysOut", _tmpKeys.get());
		_radixSortBlocksKeysOnlyKernel->setBuffer("KeysIn", keys.get());
		_radixSortBlocksKeysOnlyKernel->setUniform(_radixSortBlocksKeysOnlyKernel->uniform("nbits"), nbits);
		_radixSortBlocksKeysOnlyKernel->setUniform(_radixSortBlocksKeysOnlyKernel->uniform("numElements"), numElements);
		_radixSortBlocksKeysOnlyKernel->setUniform(_radixSortBlocksKeysOnlyKernel->uniform("startbit"), startbit);

		// Execute the compute shader
		glDispatchCompute(totalBlocks, 1, 1);

		// Insert buffer write barrier
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	void RadixSort::findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements)
	{
		Require(_findRadixOffsetsKernel, "Kernel is loaded.");

		unsigned int totalBlocks = std::max<unsigned int>(numElements / 2 / LocalSize, 1);

		// Bind the program to the pipeline
		_findRadixOffsetsKernel->bind();

		// Bind the buffers and parameters
		_findRadixOffsetsKernel->setBuffer("Counters", _counters.get());
		_findRadixOffsetsKernel->setBuffer("BlockOffsets", _blockOffsets.get());
		_findRadixOffsetsKernel->setBuffer("Keys", _tmpKeys.get());
		_findRadixOffsetsKernel->setUniform(_findRadixOffsetsKernel->uniform("startbit"), startbit);
		_findRadixOffsetsKernel->setUniform(_findRadixOffsetsKernel->uniform("numElements"), numElements);
		_findRadixOffsetsKernel->setUniform(_findRadixOffsetsKernel->uniform("totalBlocks"), totalBlocks);

		// Execute the compute shader
		glDispatchCompute(totalBlocks, 1, 1);

		// Insert buffer write barrier
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	void RadixSort::reorderDataKeysOnlyOCL(ref_ptr<Runtime::OpenGL::Buffer> keys, unsigned int startbit, unsigned int numElements)
	{
		Require(_reorderDataKeysOnlyKernel, "Kernel is loaded.");

		unsigned int totalBlocks = std::max<unsigned int>(numElements / 2 / LocalSize, 1);

		// Bind the program to the pipeline
		_reorderDataKeysOnlyKernel->bind();

		// Bind the buffers and parameters
		_reorderDataKeysOnlyKernel->setBuffer("OutKeys", keys.get());
		_reorderDataKeysOnlyKernel->setBuffer("Keys", _tmpKeys.get());
		_reorderDataKeysOnlyKernel->setBuffer("BlockOffsets", _blockOffsets.get());
		_reorderDataKeysOnlyKernel->setBuffer("Offsets", _countersSum.get());
		_reorderDataKeysOnlyKernel->setUniform(_reorderDataKeysOnlyKernel->uniform("startbit"), startbit);
		_reorderDataKeysOnlyKernel->setUniform(_reorderDataKeysOnlyKernel->uniform("numElements"), numElements);
		_reorderDataKeysOnlyKernel->setUniform(_reorderDataKeysOnlyKernel->uniform("totalBlocks"), totalBlocks);

		// Execute the compute shader
		glDispatchCompute(totalBlocks, 1, 1);

		// Insert buffer write barrier
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}
}}

#endif // VCL_OPENGL_SUPPORT
