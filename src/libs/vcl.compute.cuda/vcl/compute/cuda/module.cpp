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
#include <vcl/compute/cuda/module.h>

// C++ standard library

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace Cuda
{
	Core::owner_ptr<Module> Module::loadFromBinary(Context* ctx, const int8_t* data, size_t size)
	{
		VclRequire(ctx->isCurrent(), "Current context is set.");

		// Load the module
		CUmodule mod = 0;
		VCL_CU_SAFE_CALL(cuModuleLoadData(&mod, data));

		return Core::make_owner<Module>(mod);
	}

	Module::Module(CUmodule mod)
	: Compute::Module()
	, _module(mod)
	{
	}

	Module::~Module()
	{
		VCL_CU_SAFE_CALL(cuModuleUnload(_module));
	}

	Core::ref_ptr<Compute::Kernel> Module::kernel(const std::string& name)
	{
		auto ker = _kernels.find(name);
		if (ker != _kernels.end())
			return ker->second;

		CUresult res;
		CUfunction func;
		if ((res = cuModuleGetFunction(&func, _module, name.c_str())) == CUDA_SUCCESS)
		{
			_kernels[name] = Core::make_owner<Kernel>(name, func);
			return _kernels[name];
		} else
		{
			return nullptr;
		}
	}
}}}
