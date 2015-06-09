/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#include <vcl/compute/opencl/module.h>

// C++ standard library
#include <array>
#include <fstream>
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace OpenCL
{
	Core::owner_ptr<Module> Module::loadFromSource(Core::ref_ptr<Context> ctx, const char* source)
	{
		using namespace std;

		std::array<const char*, 1> sources = { source };
		std::array<size_t, 1> sizes = { strlen(source) + 1 };

		// Load the module
		cl_int prg_err;
		cl_program prg = clCreateProgramWithSource(*ctx, 1, sources.data(), sizes.data(), &prg_err);

		std::array<cl_device_id, 1> devices;
		devices[0] = ctx->device();
		VCL_CL_SAFE_CALL(clBuildProgram(prg, 1, devices.data(), "-cl-mad-enable", nullptr, nullptr));

		return Core::make_owner<Module>(prg);
	}

	Module::Module(cl_program mod)
	: _module(mod)
	{
	}

	Module::~Module()
	{
		VCL_CL_SAFE_CALL(clReleaseProgram(_module));
	}

	//Core::ref_ptr<Kernel> Module::kernel(const std::string& name)
	//{
	//	if (_kernels.find(name) != _kernels.end())
	//		return _kernels[name].get();
	//
	//	cl_int err;
	//	cl_kernel func = clCreateKernel(_module, name.c_str(), &err);
	//	if (err == CL_SUCCESS)
	//	{
	//		_kernels[name] = Core::make_owner<Kernel>(name, func);
	//		return _kernels[name];
	//	}
	//	else
	//	{
	//		return nullptr;
	//	}
	//}

	Module::operator cl_program () const
	{
		return _module;
	}
}}}
