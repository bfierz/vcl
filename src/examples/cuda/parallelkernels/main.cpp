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

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <iostream>

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/commandqueue.h>
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/cuda/device.h>
#include <vcl/compute/cuda/kernel.h>
#include <vcl/compute/cuda/platform.h>
#include <vcl/math/math.h>

void vectoradd
(
	cudaStream_t stream, 
	int grid_size,
	int block_size,
	int problem_size,
	const float* vecA,
	const float* vecB,
	float* vecC
);

int main(int argc, char* argv[])
{
	using namespace Vcl::Compute::Cuda;
	using Vcl::Core::ref_ptr;
	using Vcl::Core::dynamic_pointer_cast;
	using Vcl::Core::static_pointer_cast;

	const size_t problem_size = 128;

	Platform::initialise();

	for (int d = 0; d < Platform::instance()->nrDevices(); d++)
	{
		auto& dev = Platform::instance()->device(d);
		Context ctx{ dev };
		
		ref_ptr<CommandQueue> queue[] = {
			dynamic_pointer_cast<CommandQueue>(ctx.defaultQueue()),
			dynamic_pointer_cast<CommandQueue>(ctx.createCommandQueue()) };

		ref_ptr<Buffer> mem0[] = { dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, problem_size*sizeof(float))), dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, problem_size*sizeof(float))) };
		ref_ptr<Buffer> mem1[] = { dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, problem_size*sizeof(float))), dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, problem_size*sizeof(float))) };
		ref_ptr<Buffer> mem2[] = { dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, problem_size*sizeof(float))), dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, problem_size*sizeof(float))) };

		float one = 1;
		float two = 2;
		
		std::vector<float> result[2];
		result[0] = std::vector<float>(problem_size);
		result[1] = std::vector<float>(problem_size);
		for (size_t i = 0; i < 2; i++)
		{
			queue[i]->fill(static_pointer_cast<Vcl::Compute::Buffer>(mem0[i]), &one, sizeof(float));
			queue[i]->fill(static_pointer_cast<Vcl::Compute::Buffer>(mem1[i]), &two, sizeof(float));
		}
		
		for (size_t i = 0; i < 2; i++)
		{
			vectoradd(*queue[i], (int)problem_size, 32, (int)problem_size, (float*) mem0[i]->devicePtr(), (float*) mem1[i]->devicePtr(), (float*) mem2[i]->devicePtr());
			vectoradd(*queue[i], (int)problem_size, 32, (int)problem_size, (float*) mem0[i]->devicePtr(), (float*) mem1[i]->devicePtr(), (float*) mem2[i]->devicePtr());
		}
		
		for (size_t i = 0; i < 2; i++)
		{
			queue[i]->read(result[i].data(), static_pointer_cast<const Vcl::Compute::Buffer>(mem2[i]));
			queue[i]->sync();
		}
		for (auto f : result[0])
		{
			std::cout << (Vcl::Mathematics::equal(f, 300000, 1e-5f) ? '.' : 'F');
		}
		std::cout << std::endl;
		for (auto f : result[1])
		{
			std::cout << (Vcl::Mathematics::equal(f, 300000, 1e-5f) ? '.' : 'F');
		}
		std::cout << std::endl;
	}

	Platform::dispose();

	return 0;
}
