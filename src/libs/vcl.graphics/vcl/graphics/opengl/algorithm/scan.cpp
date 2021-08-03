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
#include <vcl/graphics/opengl/algorithm/scan.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/math/ceil.h>

// Compute shader
#include "scan.glslinc"
#include "scan.comp"

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics {
	namespace {
		unsigned int factorRadix2(unsigned int& log2L, unsigned int L)
		{
			if (!L)
			{
				log2L = 0;
				return 0;
			} else
			{
				for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++)
					;
				return L;
			}
		}
	}

	ScanExclusive::ScanExclusive(unsigned int maxElements)
	: _maxElements(maxElements)
	{
		using namespace Vcl::Graphics::Runtime;

		BufferDescription desc = {
			std::max(1u, maxElements / MaxWorkgroupInclusiveScanSize) * static_cast<unsigned int>(sizeof(unsigned int)),
			BufferUsage::Storage
		};

		_workSpace = make_owner<Runtime::OpenGL::Buffer>(desc);

		// Load the sorting kernels
		_scanExclusiveLocal1Kernel = Runtime::OpenGL::createComputeKernel(module, { "#define WORKGROUP_SIZE 256\n#define SCAN_SHARED_MEM_SIZE 2*WORKGROUP_SIZE\n#define scanExclusiveLocal1\n", module_scan });
		_scanExclusiveLocal2Kernel = Runtime::OpenGL::createComputeKernel(module, { "#define WORKGROUP_SIZE 256\n#define SCAN_SHARED_MEM_SIZE 2*WORKGROUP_SIZE\n#define scanExclusiveLocal2\n", module_scan });
		_uniformUpdateKernel = Runtime::OpenGL::createComputeKernel(module, { "#define WORKGROUP_SIZE 256\n#define SCAN_SHARED_MEM_SIZE 2*WORKGROUP_SIZE\n#define uniformUpdate\n", module_scan });
	}

	void ScanExclusive::operator()(
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> src,
		unsigned int arrayLength)
	{
		// Check all work-groups to be fully packed with data
		VclRequire(arrayLength % 4 == 0, "SCan works on multiles of 4");
		VclRequire(implies(arrayLength > MaxShortArraySize, arrayLength % MaxWorkgroupInclusiveScanSize == 0), "Only allow batches of full sizes.");

		if (arrayLength <= MaxShortArraySize)
		{
			scanExclusiveSmall(dst, src, 1, arrayLength);
		} else
		{
			unsigned int batchSize = arrayLength / MaxWorkgroupInclusiveScanSize;
			scanExclusiveLarge(dst, src, batchSize, MaxWorkgroupInclusiveScanSize);
		}
	}

	void ScanExclusive::scanExclusiveSmall(
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> src,
		unsigned int batchSize,
		unsigned int arrayLength)
	{
		// Check supported size range
		VclCheck((arrayLength >= MinShortArraySize) && (arrayLength <= MaxShortArraySize), "Array is within size");

		// Check total batch size limit
		VclCheck((batchSize * arrayLength) <= MaxBatchElements, "Batch size is within range");

		// Check all work-groups to be fully packed with data
		VclCheck((batchSize * arrayLength) % 4 == 0, "All work-groups are fully packed");

		return scanExclusiveLocal1(
			dst,
			src,
			batchSize,
			arrayLength);
	}

	void ScanExclusive::scanExclusiveLarge(
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> src,
		unsigned int batchSize,
		unsigned int arrayLength)
	{
		// Check power-of-two factorization
		unsigned int log2L;
		unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
		VclCheck(factorizationRemainder == 1, "Is power of two");

		// Check supported size range
		VclCheck((arrayLength >= MinLargeArraySize) && (arrayLength <= MaxLargeArraySize), "Array is within size");

		// Check total batch size limit
		VclCheck((batchSize * arrayLength) <= MaxBatchElements, "Batch size is within range");

		scanExclusiveLocal1(
			dst,
			src,
			(batchSize * arrayLength) / (4 * WorkgroupSize),
			4 * WorkgroupSize);

		scanExclusiveLocal2(
			_workSpace,
			dst,
			src,
			batchSize,
			arrayLength / (4 * WorkgroupSize));

		uniformUpdate(
			dst,
			_workSpace,
			(batchSize * arrayLength) / (4 * WorkgroupSize));
	}

	void ScanExclusive::scanExclusiveLocal1(
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> src,
		unsigned int n,
		unsigned int size)
	{
		using Vcl::Mathematics::ceil;

		VclRequire(_scanExclusiveLocal1Kernel, "Kernel is loaded.");

		// Bind the program to the pipeline
		_scanExclusiveLocal1Kernel->bind();

		// Number of elements to process
		unsigned int elements = n * size;

		// Bind the buffers and parameters
		_scanExclusiveLocal1Kernel->setBuffer("Destination", dst.get());
		_scanExclusiveLocal1Kernel->setBuffer("Source", src.get());
		_scanExclusiveLocal1Kernel->setUniform(_scanExclusiveLocal1Kernel->uniform("size"), size);
		_scanExclusiveLocal1Kernel->setUniform(_scanExclusiveLocal1Kernel->uniform("N"), elements);

		// Execute the compute shader
		unsigned int nr_workgroups = ceil((n * size) / 4, WorkgroupSize) / WorkgroupSize;
		glDispatchCompute(nr_workgroups, 1, 1);

		// Insert buffer write barrier
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	void ScanExclusive::scanExclusiveLocal2(
		ref_ptr<Runtime::OpenGL::Buffer> buffer,
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> src,
		unsigned int n,
		unsigned int size)
	{
		using Vcl::Mathematics::ceil;

		VclRequire(_scanExclusiveLocal2Kernel, "Kernel is loaded.");

		// Bind the program to the pipeline
		_scanExclusiveLocal2Kernel->bind();

		// Number of elements to process
		unsigned int elements = n * size;

		// Bind the buffers and parameters
		_scanExclusiveLocal2Kernel->setBuffer("Destination", dst.get());
		_scanExclusiveLocal2Kernel->setBuffer("Source", src.get());
		_scanExclusiveLocal2Kernel->setBuffer("Workspace", buffer.get());
		_scanExclusiveLocal2Kernel->setUniform(_scanExclusiveLocal2Kernel->uniform("N"), elements);

		// Execute the compute shader
		glDispatchCompute(ceil(elements, WorkgroupSize) / WorkgroupSize, 1, 1);

		// Insert buffer write barrier
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	void ScanExclusive::uniformUpdate(
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> buffer,
		unsigned int n)
	{
		VclRequire(_uniformUpdateKernel, "Kernel is loaded.");

		// Bind the program to the pipeline
		_uniformUpdateKernel->bind();

		// Bind the buffers and parameters
		_uniformUpdateKernel->setBuffer("Destination", dst.get());
		_uniformUpdateKernel->setBuffer("Workspace", buffer.get());

		// Execute the compute shader
		glDispatchCompute(n, 1, 1);

		// Insert buffer write barrier
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}
}}

#endif // VCL_OPENGL_SUPPORT
