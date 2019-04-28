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
#include <vcl/graphics/imageprocessing/opengl/luminance.h>

// VCL configuration
#include <vcl/config/eigen.h>

// C++ standard library
#include <tuple>

// VCL
#include <vcl/core/contract.h>

#ifdef VCL_OPENGL_SUPPORT

namespace
{
	inline std::tuple<int, int> computeNumberRequiredImages(int requested_size)
	{

		int num_images = 1;
		int max_size = 1;
		while (max_size <= requested_size)
		{
			num_images++;
			max_size *= 3;
		}

		return std::make_tuple(num_images, max_size);
	}
}

namespace Vcl { namespace Graphics { namespace ImageProcessing { namespace OpenGL
{
	Luminance::Luminance(ImageProcessor* processor)
	{
		// Kernel sources
		const char* prepare = R"(
		#version 430 core
		
		// Size of the local tile
		layout (local_size_x = 16, local_size_y = 16) in;

		// Kernel input
		layout(rgba16f, binding = 0) restrict readonly uniform image2D input0;

		// Input ranges
		uniform ivec4 inputRange0;

		// Kernel output
		layout(r16f, binding = 1) restrict writeonly uniform image2D output0;		

		// Output ranges
		uniform ivec4 outputRange0;

		 // Down scale luminance: 2x2 - Kernel
		void main()
		{
			// Ranges
			ivec2 inBase  = inputRange0.xy;
			ivec2 outBase = outputRange0.xy;
			vec2 scale = vec2(inputRange0.zw) / vec2(outputRange0.zw);

			// Compute texture coordinate of input based on region.
			ivec2 outCoords = ivec2(gl_GlobalInvocationID.xy);

			vec2 scaledCoords = vec2(outCoords) * scale;
			vec2 inFragOffset = fract(scaledCoords);
			ivec2 inCoords = ivec2(scaledCoords);

			mat2 lum;
			for (int y = 0; y <= 1; y++)
			{
				for (int x = 0; x <= 1; x++)
				{
					// Clamp the coordinates
					ivec2 inCoords = inBase + inCoords + ivec2(x, y);
					inCoords = min(inCoords, inBase + inputRange0.zw - ivec2(1, 1));

					// Compute the sum of color values
					vec4 color = imageLoad(input0, inCoords);
					lum[x][y] = dot(color, vec4(0.299f, 0.587f, 0.114f, 0.0f));
				}
			}

			float avg = dot(vec2(1.0f - inFragOffset.x, inFragOffset.x), lum * vec2(1.0f - inFragOffset.y, inFragOffset.y));
			
			// Convert to log-average
			avg = log(avg + 0.0001f);

			imageStore(output0, outBase + outCoords, vec4(avg, avg, avg, avg));
		}
		)";
		
		const char* downscale = R"(
		#version 430 core
		
		// Size of the local tile
		layout (local_size_x = 16, local_size_y = 16) in;

		// Kernel input
		layout(r16f, binding = 0) restrict readonly uniform image2D input0;

		// Input ranges
		uniform ivec4 inputRange0;

		// Kernel output
		layout(r16f, binding = 1) restrict writeonly uniform image2D output0;		

		// Output ranges
		uniform ivec4 outputRange0;

		 // Down scale luminance: 3x3 - Kernel
		void main()
		{
			// Compute texture coordinate of input based on region.
			ivec2 outCoords = ivec2(gl_GlobalInvocationID.xy);

			bool x_only = inputRange0.w == outputRange0.w;
			bool y_only = inputRange0.z == outputRange0.z;

			float avg = 0.0f;
			if (x_only)
			{
				ivec2 inCoords = 3 * outCoords + ivec2(1, 0);

				avg += imageLoad(input0, inCoords + ivec2(-1,  0)).r;
				avg += imageLoad(input0, inCoords + ivec2( 0,  0)).r;
				avg += imageLoad(input0, inCoords + ivec2( 1,  0)).r;

				avg /= 3.0f;
			}
			else if (y_only)
			{
				ivec2 inCoords = 3 * outCoords + ivec2(0, 1);

				avg += imageLoad(input0, inCoords + ivec2( 0, -1)).r;
				avg += imageLoad(input0, inCoords + ivec2( 0,  0)).r;
				avg += imageLoad(input0, inCoords + ivec2( 0,  1)).r;

				avg /= 3.0f;
			}
			else
			{
				ivec2 inCoords = 3 * outCoords + ivec2(1, 1);

				avg += imageLoad(input0, inCoords + ivec2(-1, -1)).r;
				avg += imageLoad(input0, inCoords + ivec2( 0, -1)).r;
				avg += imageLoad(input0, inCoords + ivec2( 1, -1)).r;
				avg += imageLoad(input0, inCoords + ivec2(-1,  0)).r;
				avg += imageLoad(input0, inCoords + ivec2( 0,  0)).r;
				avg += imageLoad(input0, inCoords + ivec2( 1,  0)).r;
				avg += imageLoad(input0, inCoords + ivec2(-1,  1)).r;
				avg += imageLoad(input0, inCoords + ivec2( 0,  1)).r;
				avg += imageLoad(input0, inCoords + ivec2( 1,  1)).r;

				avg /= 9.0f;
			}

			// Convert final output to a log-scale
			//if (outputRange0.z == 1 && outputRange0.w == 1)
			//{
			//	avg = log(avg + 0.0001f);
			//}

			imageStore(output0, outCoords, vec4(avg, avg, avg, avg));
		}
		)";

		_prepareKernelId = processor->buildKernel(prepare);
		_downscaleKernelId = processor->buildKernel(downscale);
	}

	void Luminance::process(ImageProcessing::ImageProcessor* processor)
	{
		VclRequire(nrInputSlots() == 1, "This kernel takes exactly one input image.");

		if (nrOutputSlots() == 0)
			return;

		// Update the used render targets
		updateResources(processor);

		// Set input to the preparation kernel
		const auto* input = inputSlot(0)->resource();

		// Find the next closest power-of-three texture with half the size
		int in_w = input->width();
		int in_h = input->height();
		int requested_out_w = (in_w + 1) / 2;
		int requested_out_h = (in_h + 1) / 2;

		// Compute number of images and maximum size
		auto intermediate_width = computeNumberRequiredImages(requested_out_w);
		auto intermediate_height = computeNumberRequiredImages(requested_out_h);
		auto num_passes = std::max(std::get<0>(intermediate_width), std::get<0>(intermediate_height));
		auto pass_width = std::get<1>(intermediate_width);
		auto pass_height = std::get<1>(intermediate_height);

		// Request an output image
		auto next_lum_base_img = processor->requestImage(pass_width, pass_height, SurfaceFormat::R16_FLOAT);
		const auto* next_lum_base_img_ptr = next_lum_base_img.get();

		Eigen::Vector4i input_range{ 0, 0, in_w, in_h };
		Eigen::Vector4i output_range{ 0, 0, pass_width, pass_height };
		
		// Convert the color image input to luminance
		processor->enqueKernel(_prepareKernelId, pass_width, pass_height, &next_lum_base_img_ptr, &output_range, 1, &input, &input_range, 1);

		// Done with the first image
		num_passes--;
		in_w = pass_width;
		in_h = pass_height;
		pass_width = std::max(pass_width / 3, 1);
		pass_height = std::max(pass_height / 3, 1);

		for (int i = 0; i < num_passes; i++)
		{
			auto curr_lum_base_img = next_lum_base_img;
			const auto* curr_lum_base_img_ptr = curr_lum_base_img.get();

			next_lum_base_img = processor->requestImage(pass_width, pass_height, SurfaceFormat::R16_FLOAT);
			next_lum_base_img_ptr = next_lum_base_img.get();

			input_range = { 0, 0, in_w, in_h };
			output_range = { 0, 0, pass_width, pass_height };

			// Convert the color image input to luminance
			processor->enqueKernel(_downscaleKernelId, pass_width, pass_height, &next_lum_base_img_ptr, &output_range, 1, &curr_lum_base_img_ptr, &input_range, 1);

			// Go to the next smaller level
			in_w = pass_width;
			in_h = pass_height;
			pass_width = std::max(pass_width / 3, 1);
			pass_height = std::max(pass_height / 3, 1);
		}

		// Set the output
		auto output = outputSlot(0);
		output->setResource(next_lum_base_img, 1, 1);
	}
}}}}

#endif // VCL_OPENGL_SUPPORT
