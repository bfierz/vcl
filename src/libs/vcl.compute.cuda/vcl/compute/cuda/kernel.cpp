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
#include <vcl/compute/cuda/kernel.h>

// VCL 
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace Cuda
{
	Kernel::Kernel(const std::string& name, CUfunction func)
	: Compute::Kernel(name)
	, _func(func)
	{
		Require(!name.empty(), "Name of CUDA function is valid.");
		Require(func != nullptr, "Pointer to CUDA function is valid.");

		_paramMemory = std::make_unique<char[]>(1024);

		VCL_CU_SAFE_CALL(cuFuncGetAttribute(&_nrMaxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, _func));
		VCL_CU_SAFE_CALL(cuFuncGetAttribute(&_staticSharedMemorySize, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, _func));
		VCL_CU_SAFE_CALL(cuFuncGetAttribute(&_constantMemorySize, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, _func));
		VCL_CU_SAFE_CALL(cuFuncGetAttribute(&_localMemorySize, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, _func));
		VCL_CU_SAFE_CALL(cuFuncGetAttribute(&_nrRegisters, CU_FUNC_ATTRIBUTE_NUM_REGS, _func));
		VCL_CU_SAFE_CALL(cuFuncGetAttribute(&_ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, _func));
		VCL_CU_SAFE_CALL(cuFuncGetAttribute(&_binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, _func));
	}

	void Kernel::setCacheConfiguration(CacheConfig config)
	{
		CUfunc_cache cache = CU_FUNC_CACHE_PREFER_NONE;
		switch (config)
		{
		case CacheConfig::PreferNone:
			cache = CU_FUNC_CACHE_PREFER_NONE;
			break;
		case CacheConfig::PreferSharedMemory:
			cache = CU_FUNC_CACHE_PREFER_SHARED;
			break;
		case CacheConfig::PreferL1:
			cache = CU_FUNC_CACHE_PREFER_L1;
			break;
		}

		VCL_CU_SAFE_CALL(cuFuncSetCacheConfig(_func, cache));
	}

	void Kernel::run(CommandQueue& queue, dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory)
	{
		runImpl(queue, gridDim, blockDim, dynamicSharedMemory, nullptr);
	}

	void Kernel::runImpl(CommandQueue& queue, dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, void** params)
	{
		VCL_CU_SAFE_CALL(cuLaunchKernel
		(
			_func,
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			dynamicSharedMemory,
			(CUstream) queue,
			nullptr, params
		));
#ifdef VCL_DEBUG
		queue.sync();
#endif
	}
}}}
