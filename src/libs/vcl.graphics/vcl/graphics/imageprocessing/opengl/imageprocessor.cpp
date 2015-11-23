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
#include <vcl/graphics/imageprocessing/opengl/imageprocessor.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/opengl/resource/shader.h>
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/util/hashedstring.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing { namespace OpenGL
{
	size_t ImageProcessor::buildKernel(const char* source)
	{
		using Runtime::OpenGL::Shader;
		using Runtime::OpenGL::ShaderProgramDescription;
		using Runtime::OpenGL::ShaderProgram;

		size_t kernelId = Util::StringHash{ source }.hash();
		Shader kernel{ Runtime::ShaderType::ComputeShader, 0, source };

		ShaderProgramDescription desc;
		desc.ComputeShader = &kernel;
		_kernel.emplace(kernelId, std::make_unique<ShaderProgram>(desc));

		return kernelId;
	}

	ImageProcessor::ImagePtr ImageProcessor::requestImage(int w, int h, SurfaceFormat fmt)
	{
		auto& cache = _textures[fmt];
		auto cache_entry = std::find_if(cache.begin(), cache.end(), [w, h](const ImagePtr& img)
		{
			return img->width() < w && img->height() < h;
		});
		if (cache_entry != cache.end())
		{
			return *cache_entry;
		}
		else
		{
			Runtime::Texture2DDescription desc2d;
			desc2d.Format = fmt;
			desc2d.ArraySize = 1;
			desc2d.Width = w;
			desc2d.Height = h;
			desc2d.MipLevels = 1;
			auto new_cache_entry = std::make_shared<Runtime::OpenGL::Texture2D>(desc2d);
			cache.push_back(new_cache_entry);

			return new_cache_entry;
		}
	}

	void ImageProcessor::enqueKernel
	(
		size_t kernel,
		const Runtime::Texture** outputs, Eigen::Vector4i* outRanges, size_t nr_outputs,
		const Runtime::Texture** inputs, Eigen::Vector4i* inRanges, size_t nr_inputs
	)
	{
		Require(nr_inputs <= 8, "Supports 8 input slots");

		// Early out
		if (nr_outputs == 0)
			return;

		// Fetch the program
		auto ker = _kernel.find(kernel);
		if (ker == _kernel.end())
			return;

		auto prog = ker->second.get();

		// Bind the program to the pipeline
		prog->bind();

		// Bind the input
		char input_name[] = "input0";
		char input_range_name[] = "inputRange0";
		for (int i = 0; i < nr_inputs; i++)
		{
			input_name[5] = '0' + i;
			input_range_name[10] = '0' + i;

			auto in_handle = prog->uniform(input_name);
			prog->setImage(in_handle, inputs[i], true, false);

			auto in_range_handle = prog->uniform(input_range_name);
			prog->setUniform(in_range_handle, Eigen::Vector4i{ inRanges[i].x(), inRanges[i].y(), inRanges[i].z(), inRanges[i].w() });
		}

		// Bind the output parameter
		auto out_handle = prog->uniform("output0");
		prog->setImage(out_handle, outputs[0], false, true);

		auto out_range_handle = prog->uniform("outputRange0");
		prog->setUniform(out_range_handle, Eigen::Vector4i{ outRanges[0].x(), outRanges[0].y(), outRanges[0].z(), outRanges[0].w() });

		// Execute the compute shader
		int w = outRanges[0].z();
		int h = outRanges[0].w();
		glDispatchCompute(w, h, 1);
	}
}}}}
