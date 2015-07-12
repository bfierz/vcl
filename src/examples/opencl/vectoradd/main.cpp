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

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <iostream>

// VCL
#include <vcl/compute/opencl/buffer.h>
#include <vcl/compute/opencl/context.h>
#include <vcl/compute/opencl/device.h>
#include <vcl/compute/opencl/kernel.h>
#include <vcl/compute/opencl/platform.h>
#include <vcl/math/math.h>

extern uint32_t vectoradd[];

int main(int argc, char* argv[])
{
	using namespace Vcl::Compute::OpenCL;

	Platform::initialise();

	for (int d = 0; d < Platform::instance()->nrDevices(); d++)
	{
		auto& dev = Platform::instance()->device(d);
		auto ctx = Context{ dev };

		auto queue = Vcl::Core::dynamic_pointer_cast<CommandQueue>(ctx.defaultQueue());

		auto mem0 = Vcl::Core::dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 1024*sizeof(float)));
		auto mem1 = Vcl::Core::dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 1024*sizeof(float)));
		auto mem2 = Vcl::Core::dynamic_pointer_cast<Buffer>(ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 1024*sizeof(float)));

		float one = 1;
		float two = 2;
		queue->fill(*mem0, &one, sizeof(float));
		queue->fill(*mem1, &two, sizeof(float));

		auto mod = ctx.createModuleFromSource((const char*) vectoradd);
		auto kernel = Vcl::Core::dynamic_pointer_cast<Kernel>(mod->kernel("vectoradd"));
		kernel->run(*queue, 1, { 1024, 0, 0 }, { 128, 0, 0 }, (cl_mem) *mem0, (cl_mem) *mem1, (cl_mem) *mem2);

		std::vector<float> result(1024);
		queue->read(result.data(), *mem2);
		queue->sync();

		for (auto f : result)
		{
			std::cout << (Vcl::Mathematics::equal(f, 3, 1e-5f) ? '.' : 'F');
		}
		std::cout << std::endl;
	}

	Platform::dispose();

	return 0;
}
