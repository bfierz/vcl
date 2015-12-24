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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/cuda.h>

// C++ standard library
#include <array>
#include <string>

// CUDA
#include <vector_types.h>

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/commandqueue.h>
#include <vcl/compute/kernel.h>

namespace Vcl { namespace Compute { namespace Cuda
{
	template<typename T>
	struct KernelArg
	{
		KernelArg(const T& arg) : Arg(arg) { /*static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "T is simple type.");*/ }

		static size_t size() { return sizeof(T); }
		 void* ptr() { return &Arg; }

	private:
		T Arg;
	};

	template<>
	struct KernelArg<dim3>
	{
		KernelArg(const dim3& arg) : Arg(arg) {}

		static size_t size() { return sizeof(dim3); }
		void* ptr() { return &Arg; }

	private:
		dim3 Arg;
	};
	template<typename U>
	struct KernelArg<ref_ptr<U>>
	{
		KernelArg(const ref_ptr<U>& arg) : Arg((CUdeviceptr) *arg) { static_assert(std::is_base_of<Compute::Buffer, std::decay<U>::type>::value, "Type is derived from Buffer."); }

		static size_t size() { return sizeof(CUdeviceptr); }
		void* ptr() { return &Arg; }

	private:
		CUdeviceptr Arg;
	};
	template<>
	struct KernelArg<Cuda::Buffer>
	{
		KernelArg(const Cuda::Buffer& arg) : Arg((CUdeviceptr) arg) {}

		static size_t size() { return sizeof(CUdeviceptr); }
		void* ptr() { return &Arg; }

	private:
		CUdeviceptr Arg;
	};

	class Kernel : public Compute::Kernel
	{
		enum class CacheConfig
		{
			PreferNone,
			PreferSharedMemory,
			PreferL1
		};

	public:
		Kernel(const std::string& name, CUfunction func);
		virtual ~Kernel() = default;

	public: // Attributes
		int nrMaxThreadsPerBlock() const { return _nrMaxThreadsPerBlock; }
		int sharedMemorySize() const { return _staticSharedMemorySize; }
		int constantMemorySize() const { return _constantMemorySize; }
		int localMemorySize() const { return _localMemorySize; }
		int nrRegisters() const { return _nrRegisters; }
		int ptxVersion() const { return _ptxVersion; }
		int binaryVersion() const { return _binaryVersion; }

	public: // Kernel configuration
		void setCacheConfiguration(CacheConfig config);

	public:
		template<typename... Args>
		void run
		(
			CommandQueue& queue, dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory,
			const Args&... args
		)
		{
			void* params [] = { KernelArg<typename std::decay<Args>::type> {args}.ptr()... };
			//void* params [] = { ((void*) &args)... };
			
			runImpl(queue, gridDim, blockDim, dynamicSharedMemory, params);
		}

		void run(CommandQueue& queue, dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory);
		
	private:
		void runImpl(CommandQueue& queue, dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, void** params);

	private: // Kernel data

		//! Pointer to the device kernel
		CUfunction _func;

	private: // Kernel information
		int _nrMaxThreadsPerBlock;
		int _staticSharedMemorySize;
		int _constantMemorySize;
		int _localMemorySize;
		int _nrRegisters;
		int _ptxVersion;
		int _binaryVersion;
	};
}}}
