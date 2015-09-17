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
#include <vcl/config/opencl.h>

// C++ standard library
#include <array>
#include <string>

// VCL
#include <vcl/compute/opencl/commandqueue.h>
#include <vcl/compute/kernel.h>

namespace Vcl { namespace Compute { namespace OpenCL
{
	struct LocalMemory
	{
		LocalMemory(size_t size) : Size(size) {}

		size_t Size;
	};

	template<typename T>
	struct KernelArg
	{
		static size_t size(const T&) { return sizeof(T); }
		static const void* ptr(const T& arg) { return &arg; }
	};

	template<>
	struct KernelArg<LocalMemory>
	{
		static size_t size(const LocalMemory& arg) { return arg.Size; }
		static const void* ptr(const LocalMemory&) { return nullptr; }
	};

	class Kernel : public Compute::Kernel
	{
	public:
		Kernel(const std::string& name, cl_kernel func);
		virtual ~Kernel() = default;
		
	public:
		template<typename... Args>
		void run
		(
			CommandQueue& queue, int dim, std::array<size_t, 3> globalDim, std::array<size_t, 3> localDim,
			const Args&... args
		)
		{
			pushArgs<0>(args...);

			run(queue, dim, globalDim, localDim);
		}

		void run(CommandQueue& queue, int dim, std::array<size_t, 3> globalDim, std::array<size_t, 3> localDim);

	private:
		template<int I, typename Arg, typename... Args>
		void pushArgs(const Arg& arg, const Args&... args)
		{
			VCL_CL_SAFE_CALL(clSetKernelArg(_func, I, KernelArg<Arg>::size(arg), KernelArg<Arg>::ptr(arg)));

			pushArgs<I + 1>(args...);
		}

		template<int I, typename Arg>
		void pushArgs(const Arg& arg)
		{
			VCL_CL_SAFE_CALL(clSetKernelArg(_func, I, KernelArg<Arg>::size(arg), KernelArg<Arg>::ptr(arg)));
		}

	private:
		cl_kernel _func;
	};
}}}
