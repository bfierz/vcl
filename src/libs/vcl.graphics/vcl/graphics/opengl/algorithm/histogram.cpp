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
#include <vcl/graphics/opengl/algorithm/histogram.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/math/ceil.h>

// Format library
#include <fmt/format.h>

// Compute shader
#include "histogram.comp"

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics
{
	Histogram::Histogram(unsigned int nr_elements, unsigned int nr_buckets)
	: _maxNrElements(nr_elements)
	, _maxNrBuckets(nr_buckets)
	{
		using namespace Vcl::Graphics::Runtime;
		using Vcl::Mathematics::ceil;

		VclRequire(nr_buckets < 2048, "Shared memory buckets are limited to 2048.");

		unsigned int totalBlocks = ceil(_maxNrElements, LocalSize) / LocalSize;
		BufferDescription desc =
		{
			totalBlocks * nr_buckets * static_cast<unsigned int>(sizeof(unsigned int)),
			BufferUsage::Storage
		};
		
		_partialHistograms = make_owner<Runtime::OpenGL::Buffer>(desc);

		// Define the number of buckets for the shader
		auto partials = fmt::format("#define partialHistograms\n#define NUM_BUCKETS {}u", nr_buckets);
		auto collect  = fmt::format("#define collectPartialHistograms\n#define NUM_BUCKETS {}u", nr_buckets);

		// Load the kernels
		_partialHistogramKernel         = Runtime::OpenGL::createComputeKernel(module, { partials.c_str() });
		_collectPartialHistogramsKernel = Runtime::OpenGL::createComputeKernel(module, { collect.c_str()  });
	}

	void Histogram::operator()
	(
		ref_ptr<Runtime::OpenGL::Buffer> histogram,
		const ref_ptr<Runtime::OpenGL::Buffer> values,
		unsigned int num_elements
	)
	{
		// Build the histogram per work-group
		partialHistograms(_partialHistograms, values, num_elements, _maxNrBuckets);

		// Collect the previous partial histograms
		collectPartialHistograms(histogram, _partialHistograms, num_elements, _maxNrBuckets);
	}

	void Histogram::partialHistograms
	(
		ref_ptr<Runtime::OpenGL::Buffer> buckets,
		const ref_ptr<Runtime::OpenGL::Buffer> values,
		unsigned int num_elements,
		unsigned int num_buckets
	)
	{
		using Vcl::Mathematics::ceil;

		VclRequire(_partialHistogramKernel, "Kernel is loaded.");

		unsigned int totalBlocks = ceil(num_elements, LocalSize) / LocalSize;

		// Bind the program to the pipeline
		_partialHistogramKernel->bind();

		// Bind the buffers and parameters
		_partialHistogramKernel->setBuffer("Buckets", buckets.get());
		_partialHistogramKernel->setBuffer("Values", values.get());
		_partialHistogramKernel->setUniform(_partialHistogramKernel->uniform("numElements"), num_elements);
		_partialHistogramKernel->setUniform(_partialHistogramKernel->uniform("numBuckets"), num_buckets);

		// Execute the compute shader
		glDispatchCompute(totalBlocks, 1, 1);

		// Insert buffer write barrier
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	void Histogram::collectPartialHistograms
	(
		ref_ptr<Runtime::OpenGL::Buffer> histogram,
		const ref_ptr<Runtime::OpenGL::Buffer> buckets,
		unsigned int num_elements,
		unsigned int num_buckets
	)
	{
		using Vcl::Mathematics::ceil;

		VclRequire(_partialHistogramKernel, "Kernel is loaded.");

		unsigned int num_partial_histograms = ceil(num_elements, LocalSize) / LocalSize;
		unsigned int totalBlocks = ceil(num_buckets, LocalSize) / LocalSize;

		// Bind the program to the pipeline
		_collectPartialHistogramsKernel->bind();

		// Bind the buffers and parameters
		_collectPartialHistogramsKernel->setBuffer("Histogram", histogram.get());
		_collectPartialHistogramsKernel->setBuffer("Buckets", buckets.get());
		_collectPartialHistogramsKernel->setUniform(_collectPartialHistogramsKernel->uniform("numBuckets"), num_buckets);
		_collectPartialHistogramsKernel->setUniform(_collectPartialHistogramsKernel->uniform("numPartialHistograms"), num_partial_histograms);

		// Execute the compute shader
		glDispatchCompute(totalBlocks, 1, 1);

		// Insert buffer write barrier
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}
}}

#endif // VCL_OPENGL_SUPPORT
