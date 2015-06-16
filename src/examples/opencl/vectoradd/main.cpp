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
#include <vcl/compute/opencl/context.h>
#include <vcl/compute/opencl/device.h>
#include <vcl/compute/opencl/kernel.h>
#include <vcl/compute/opencl/platform.h>

extern uint32_t vectoradd[];

int main(int argc, char* argv[])
{
	using namespace Vcl::Compute::OpenCL;

	Platform::initialise();

	for (int d = 0; d < Platform::instance()->nrDevices(); d++)
	{
		auto& dev = Platform::instance()->device(d);
		auto ctx = Context{ dev };

		auto mem0 = ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 1024);
		auto mem1 = ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 1024);
		auto mem2 = ctx.createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 1024);

		auto mod = ctx.createModuleFromSource((const char*) vectoradd);
		auto kernel = Vcl::Core::dynamic_pointer_cast<Kernel>(mod->kernel("vectoradd"));
		kernel->run();
	}

	Platform::dispose();

	return 0;
}
