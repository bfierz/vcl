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
#include <vcl/graphics/imageprocessing/opengl/gaussian.h>

// VCL
#include <vcl/core/contract.h>

// Local
#include "3rdparty/GaussianBlur.h"

namespace Vcl { namespace Graphics { namespace ImageProcessing { namespace OpenGL {
	Gaussian::Gaussian(ImageProcessor* processor)
	{
		// Various Kernels from http://dev.theomader.com/gaussian-kernel-calculator/
		// Sigma: 1
		// * 0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f
		// * 0.00598f, 0.060626f, 0.241843f, 0.383103f, 0.241843f, 0.060626f, 0.00598f
		// * 0.000229f, 0.005977f, 0.060598f, 0.241732f, 0.382928f, 0.241732f, 0.060598f, 0.005977f, 0.000229f
		// Sigma: 2
		// * 0.071303f, 0.131514f, 0.189879f, 0.214607f, 0.189879f, 0.131514f, 0.071303f
		// * 0.028532f, 0.067234f, 0.124009f, 0.179044f, 0.20236f, 0.179044f, 0.124009f, 0.067234f, 0.028532f
		// Sigma: 3
		// * 0.063327f, 0.093095f, 0.122589f, 0.144599f, 0.152781f, 0.144599f, 0.122589f, 0.093095f, 0.063327f
		// * 0.035822f, 0.05879f, 0.086425f, 0.113806f, 0.13424f, 0.141836f, 0.13424f, 0.113806f, 0.086425f, 0.05879f, 0.035822f
		// Sigma: 5
		// * 0.066414f, 0.079465f, 0.091364f, 0.100939f, 0.107159f, 0.109317f, 0.107159f, 0.100939f, 0.091364f, 0.079465f, 0.066414f

		// Kernel source
		const char* h = R"(
		#version 430 core

		// Size of the local tile
		layout (local_size_x = 16, local_size_y = 16) in;

		// Kernel input
		layout(rgba16f, binding = 0) restrict readonly uniform image2D input0;

		// Input ranges
		uniform ivec4 inputRange0;

		// Kernel output
		layout(rgba16f, binding = 1) restrict writeonly uniform image2D output0;		

		// Output ranges
		uniform ivec4 outputRange0;

		// Kernel weights
		const float w[9] = { 0.000229f, 0.005977f, 0.060598f, 0.241732f, 0.382928f, 0.241732f, 0.060598f, 0.005977f, 0.000229f };

		void main()
		{
			ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

			int o0 = coords.x >= 4 ? -4 : 4;
			int o1 = coords.x >= 3 ? -3 : 3;
			int o2 = coords.x >= 2 ? -2 : 2;
			int o3 = coords.x >= 1 ? -1 : 1;

			int o5 = coords.x + 1 < outputRange0.z ? 1 : -1;
			int o6 = coords.x + 2 < outputRange0.z ? 2 : -2;
			int o7 = coords.x + 3 < outputRange0.z ? 3 : -3;
			int o8 = coords.x + 4 < outputRange0.z ? 4 : -4;

			vec4 c0 = imageLoad(input0, coords + ivec2(o0, 0));
			vec4 c1 = imageLoad(input0, coords + ivec2(o1, 0));
			vec4 c2 = imageLoad(input0, coords + ivec2(o2, 0));
			vec4 c3 = imageLoad(input0, coords + ivec2(o3, 0));
			vec4 c4 = imageLoad(input0, coords + ivec2( 0, 0));
			vec4 c5 = imageLoad(input0, coords + ivec2(o5, 0));
			vec4 c6 = imageLoad(input0, coords + ivec2(o6, 0));
			vec4 c7 = imageLoad(input0, coords + ivec2(o7, 0));
			vec4 c8 = imageLoad(input0, coords + ivec2(o8, 0));

			vec4 blurred =
				c0*w[0] + c1*w[1] + c2*w[2] + c3*w[3] +
				c4*w[4] + 
				c5*w[5] + c6*w[6] + c7*w[7] + c8*w[8];

			imageStore(output0, coords, blurred);
		}
		)";

		const char* v = R"(
		#version 430 core

		// Size of the local tile
		layout (local_size_x = 16, local_size_y = 16) in;

		// Kernel input
		layout(rgba16f, binding = 0) restrict readonly uniform image2D input0;

		// Input ranges
		uniform ivec4 inputRange0;

		// Kernel output
		layout(rgba16f, binding = 1) restrict writeonly uniform image2D output0;		

		// Output ranges
		uniform ivec4 outputRange0;

		// Kernel weights
		const float w[9] = { 0.000229f, 0.005977f, 0.060598f, 0.241732f, 0.382928f, 0.241732f, 0.060598f, 0.005977f, 0.000229f };

		void main()
		{
			ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

			int o0 = coords.y >= 4 ? -4 : 4;
			int o1 = coords.y >= 3 ? -3 : 3;
			int o2 = coords.y >= 2 ? -2 : 2;
			int o3 = coords.y >= 1 ? -1 : 1;

			int o5 = coords.y + 1 < outputRange0.w ? 1 : -1;
			int o6 = coords.y + 2 < outputRange0.w ? 2 : -2;
			int o7 = coords.y + 3 < outputRange0.w ? 3 : -3;
			int o8 = coords.y + 4 < outputRange0.w ? 4 : -4;

			vec4 c0 = imageLoad(input0, coords + ivec2(0, o0));
			vec4 c1 = imageLoad(input0, coords + ivec2(0, o1));
			vec4 c2 = imageLoad(input0, coords + ivec2(0, o2));
			vec4 c3 = imageLoad(input0, coords + ivec2(0, o3));
			vec4 c4 = imageLoad(input0, coords + ivec2(0,  0));
			vec4 c5 = imageLoad(input0, coords + ivec2(0, o5));
			vec4 c6 = imageLoad(input0, coords + ivec2(0, o6));
			vec4 c7 = imageLoad(input0, coords + ivec2(0, o7));
			vec4 c8 = imageLoad(input0, coords + ivec2(0, o8));

			vec4 blurred =
				c0*w[0] + c1*w[1] + c2*w[2] + c3*w[3] +
				c4*w[4] + 
				c5*w[5] + c6*w[6] + c7*w[7] + c8*w[8];

			imageStore(output0, coords, blurred);
		}
		)";

		_horizontalKernelId = processor->buildKernel(h);
		_verticalKernelId = processor->buildKernel(v);
	}

	void Gaussian::process(ImageProcessing::ImageProcessor* processor)
	{
		if (nrInputSlots() == 0 || nrOutputSlots() == 0)
			return;

		// Update the used render targets
		updateResources(processor);

		// Configuration
		size_t nr_outputs = 1;
		size_t nr_inputs = 1;
		Eigen::Vector4i output_range{ 0, 0, _outputSlots[0]->resource()->width(), _outputSlots[0]->resource()->height() };
		Eigen::Vector4i input_range{ 0, 0, _inputSlots[0]->resource()->width(), _inputSlots[0]->resource()->height() };

		// Temporary storage
		auto tmp = processor->requestImage(output_range.z(), output_range.w(), _inputSlots[0]->resource()->format());

		// Execute the horizontal blurring kernel
		const Runtime::Texture* houtput = tmp.get();
		const Runtime::Texture* hinput = _inputSlots[0]->resource();

		processor->enqueKernel(_horizontalKernelId, output_range.z(), output_range.w(), &houtput, &output_range, nr_outputs, &hinput, &input_range, nr_inputs);

		// Execute the vertical blurring kernel
		const Runtime::Texture* voutput = _outputSlots[0]->resource();
		const Runtime::Texture* vinput = tmp.get();

		processor->enqueKernel(_verticalKernelId, output_range.z(), output_range.w(), &voutput, &output_range, nr_outputs, &vinput, &input_range, nr_inputs);
	}
}}}}
