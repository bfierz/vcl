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

// C++ standard library
#include <string>
#include <unordered_map>

// VCL 
#include <vcl/graphics/imageprocessing/imageprocessor.h>
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing { namespace OpenGL
{
	class ImageProcessor : public ImageProcessing::ImageProcessor
	{
	public:
		size_t buildKernel(const char* source);

	public:
		virtual ImagePtr requestImage(int w, int h, SurfaceFormat fmt) override;
		virtual void enqueKernel
		(
			size_t kernel,
			const Runtime::Texture** outputs, Eigen::Vector4i* outRanges, size_t nr_outputs,
			const Runtime::Texture** inputs, Eigen::Vector4i* inRanges, size_t nr_inputs
		) override;

	private:
		//! Kernel cache
		std::unordered_map<size_t, std::unique_ptr<Runtime::OpenGL::ShaderProgram>> _kernel;

		//! Texture cache
		std::unordered_map<SurfaceFormat, std::vector<ImagePtr>> _textures;
	};
}}}}
