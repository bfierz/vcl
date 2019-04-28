/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#include <vcl/graphics/imageprocessing/opengl/conversion.h>

// VCL
#include <vcl/core/contract.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace ImageProcessing { namespace OpenGL
{
	IntegerConversion::IntegerConversion(ImageProcessor* processor)
	{
		// Kernel source
		const char* cs = R"(
		#version 430 core

		// Size of the local tile
		layout (local_size_x = 16, local_size_y = 16) in;

		// Kernel input
		layout(rgba32i, binding = 0) restrict readonly uniform image2D input0;

		// Input ranges
		uniform ivec4 inputRange0;

		// Kernel output
		layout(binding = 1) restrict writeonly uniform image2D output0;		

		// Output ranges
		uniform ivec4 outputRange0;

		// Min-max
		uniform ivec2 conversionRange;

		void main()
		{
			ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
			vec4 values = imageLoad(input0, coords);

			vec4 normalized_ext;
			if (conversionRange.y - conversionRange.x < (1 << 31 - 1))
			{
				normalized_ext = (values - vec4(conversionRange.x)) / vec4(conversionRange.y - conversionRange.x);
			}
			else
			{
			}

			imageStore(output0, coords, normalized_ext * 0.5f + vec4(0.5f));
		}
		)";

		_kernelId = processor->buildKernel(cs);
	}

	void IntegerConversion::process(ImageProcessing::ImageProcessor* processor)
	{
		if (nrInputSlots() == 0 || nrOutputSlots() == 0)
			return;

		// Update the used render targets
		updateResources(processor);

		// Execute the image kernel
		size_t nr_outputs = 1;
		const Runtime::Texture* output = _outputSlots[0]->resource();
		Eigen::Vector4i output_range{ 0, 0, output->width(), output->height() };

		size_t nr_inputs = 1;
		const Runtime::Texture* input = _inputSlots[0]->resource();
		Eigen::Vector4i input_range{ 0, 0, input->width(), input->height() };

		processor->enqueKernel(_kernelId, output->width(), output->height(), &output, &output_range, nr_outputs, &input, &input_range, nr_inputs);
	}
}}}}

#endif // VCL_OPENGL_SUPPORT
