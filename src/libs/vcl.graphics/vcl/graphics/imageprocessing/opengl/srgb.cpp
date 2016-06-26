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
#include <vcl/graphics/imageprocessing/opengl/srgb.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing { namespace OpenGL
{
	SRGB::SRGB(ImageProcessor* processor)
	{
		// Kernel source
		const char* cs = R"(
		#version 430 core

		// Size of the local tile
		layout (local_size_x = 16, local_size_y = 16) in;

		// Kernel input
		layout(rgba8) uniform image2D input0;

		// Input ranges
		uniform uvec4 inputRange0;

		// Kernel output
		layout(rgba8) uniform image2D output0;		

		// Output ranges
		uniform uvec4 outputRange0;

		float LinearToSrgb(float val)
		{
			float ret;
			if (val <= 0.0)
				ret = 0.0f;
			else if (val <= 0.0031308f)
				ret = 12.92f*val;
			else if (val <= 1.0f)
				ret = (pow(val, 1.0f/2.4f) * 1.055f) - 0.055f;
			else
				ret = 1.0f;
			return ret;
		}

		void main()
		{
			ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
			vec4 values = imageLoad(input0, coords);

			float sr = LinearToSrgb(values.r);
			float sg = LinearToSrgb(values.g);
			float sb = LinearToSrgb(values.b);
			float sa = values.a;

			imageStore(output0, coords, vec4(sr, sg, sb, sa));
		}
		)";

		_kernelId = processor->buildKernel(cs);
	}

	void SRGB::process(ImageProcessing::ImageProcessor* processor)
	{
		if (nrOutputSlots() == 0)
			return;

		// Update the used render targets
		updateResources(processor);

		// Let the concrete implementation set its own parameters
		setTaskParameters(processor);

		// Execute the image kernel
		size_t nr_outputs = _outputSlots.size();
		const OutputSlot* outputs[16];
		for (int i = 0; i < nr_outputs; i++)
			outputs[i] = _outputSlots[i].get();

		size_t nr_inputs = _inputSlots.size();
		const InputSlot* inputs[16];
		for (int i = 0; i < nr_inputs; i++)
			inputs[i] = _inputSlots[i].get();

		processor->enqueKernel(_kernelId, outputs, nr_outputs, inputs, nr_inputs);
	}
}}}}
