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
#include <vcl/graphics/imageprocessing/opengl/tonemap.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing { namespace OpenGL
{
	Tonemap::Tonemap(ImageProcessor* processor)
	{
		// Kernel source
		const char* reinhard = R"(
		#version 430 core

		// Size of the local tile
		layout (local_size_x = 16, local_size_y = 16) in;

		// Kernel input
		// * Input[0] -> Rendered scene
		// * Input[1] -> Average luminance
		layout(rgba16f) restrict readonly uniform image2D input0;
		layout(r16f)    restrict readonly uniform image2D input1;

		// Input ranges
		uniform uvec4 inputRange0;
		uniform uvec4 inputRange1;

		// Kernel output
		restrict writeonly uniform image2D output0;		

		// Output ranges
		uniform uvec4 outputRange0;
		
		// Approximates luminance from an RGB value
		float CalcLuminance(vec3 color)
		{
			return max(dot(color, vec3(0.299f, 0.587f, 0.114f)), 0.0001f);
		}
		
		// Retrieves the log-average luminance from the texture
		float GetAvgLuminance()
		{
			return exp(imageLoad(input1, ivec2(0, 0)).r);
		}
		
		// Applies the filmic curve from John Hable's presentation
		vec3 ToneMapFilmicALU(vec3 colour)
		{
			colour = max(vec3(0.0f), colour - vec3(0.004f));
			colour = (colour * (6.2f * colour + 0.5f)) / (colour * (6.2f * colour + 1.7f)+ 0.06f);
		
			// result has 1/2.2 baked in
			// return pow(colour, vec3(2.2f, 2.2f, 2.2f));
			return colour;
		}
		
		// Determines the colour based on exposure settings
		vec3 CalcExposedColor(vec3 colour, float avgLuminance, float threshold, out float exposure)
		{
			// Use geometric mean
			avgLuminance = max(avgLuminance, 0.001f);
			float keyValue = 1.03f - 2.0f / (2.0f + log(avgLuminance + 1.0f)/log(10.0f));
			float linearExposure = (keyValue / avgLuminance);
			exposure = log2(max(linearExposure, 0.0001f));
			exposure -= threshold;
			return exp2(exposure) * colour;
		}
		
		// Applies exposure and tone mapping to the specific color, and applies
		// the threshold to the exposure value.
		vec3 ToneMap(vec3 colour, float avgLuminance, float threshold, out float exposure)
		{
			float pixelLuminance = CalcLuminance(colour);
			colour = CalcExposedColor(colour, avgLuminance, threshold, exposure);
			colour = ToneMapFilmicALU(colour);
			return colour;
		}
		
		void main()
		{
			ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
			vec4 colour = imageLoad(input0, coords);
			float avg_lum = GetAvgLuminance();

			// Tone map the primary input
			float exposure = 0;

			vec3 rgb = ToneMap(colour.rgb, avg_lum, 0, exposure);
			float a = colour.a;

			imageStore(output0, coords, vec4(rgb, a));
		}
		)";

		_reinhardKernelId = processor->buildKernel(reinhard);
	}

	void Tonemap::process(ImageProcessing::ImageProcessor* processor)
	{
		if (nrInputSlots() == 0 || nrOutputSlots() == 0)
			return;

		// Update the used render targets
		updateResources(processor);

		// Execute the image kernel
		size_t nr_outputs = 1;
		const Runtime::Texture* output = _outputSlots[0]->resource();
		Eigen::Vector4i output_range{ 0, 0, output->width(), output->height() };

		size_t nr_inputs = 2;
		const Runtime::Texture* inputs[] = { _inputSlots[0]->resource(), _inputSlots[1]->resource() };
		Eigen::Vector4i input_ranges[] = 
		{
			{ 0, 0, inputs[0]->width(), inputs[0]->height() },
			{ 0, 0, inputs[1]->width(), inputs[1]->height() }
		};

		processor->enqueKernel(_reinhardKernelId, output->width(), output->height(), &output, &output_range, nr_outputs, inputs, input_ranges, nr_inputs);
	}
}}}}
