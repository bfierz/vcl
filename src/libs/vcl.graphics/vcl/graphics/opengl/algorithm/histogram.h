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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics
{
	class Histogram
	{
	public:
		Histogram(unsigned int nr_elements, unsigned int nr_buckets);
		virtual ~Histogram() = default;

	public:
		void operator()
		(
			ref_ptr<Runtime::OpenGL::Buffer> histogram,
			const ref_ptr<Runtime::OpenGL::Buffer> values,
			unsigned int num_elements
		);

	private:

		void partialHistograms
		(
			ref_ptr<Runtime::OpenGL::Buffer> buckets,
			const ref_ptr<Runtime::OpenGL::Buffer> values,
			unsigned int num_elements,
			unsigned int num_buckets
		);

		void collectPartialHistograms
		(
			ref_ptr<Runtime::OpenGL::Buffer> histogram,
			const ref_ptr<Runtime::OpenGL::Buffer> buckets,
			unsigned int num_elements,
			unsigned int num_buckets
		);

	private: // Limits

		//! Warp size
		static const unsigned int WarpSize = 32;

		//! Work group size
		static const unsigned int LocalSize = 128;

	private: // Configuration

		unsigned int _maxNrElements{ 0 };

		unsigned int _maxNrBuckets{ 0 };

	private: // Module, Kernels

		std::unique_ptr<Runtime::OpenGL::ShaderProgram> _partialHistogramKernel;
		std::unique_ptr<Runtime::OpenGL::ShaderProgram> _collectPartialHistogramsKernel;

	private: // Buffers

		//! Buffer accumulating the partial results
		owner_ptr<Runtime::OpenGL::Buffer> _partialHistograms;
	};
}}

#endif // VCL_OPENGL_SUPPORT
