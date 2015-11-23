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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <memory>
#include <string>

// VCL 
#include <vcl/graphics/imageprocessing/task.h>
#include <vcl/graphics/runtime/resource/texture.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing
{
	class ImageProcessor
	{
	public:
		using ImagePtr = std::shared_ptr<Runtime::Texture>;

	public:
		void execute(Task* filter);

	public:
		virtual void enqueKernel
		(
			size_t kernel,
			const Runtime::Texture** outputs, Eigen::Vector4i* outRanges, size_t nr_outputs,
			const Runtime::Texture** inputs, Eigen::Vector4i* inRanges, size_t nr_inputs
		) = 0;

	public: // Resource management
		virtual ImagePtr requestImage(int w, int h, SurfaceFormat fmt) = 0;
	};
}}}
