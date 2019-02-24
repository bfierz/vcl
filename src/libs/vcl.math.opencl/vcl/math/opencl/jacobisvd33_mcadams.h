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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>
#include <vcl/config/opencl.h>

// VCL
#include <vcl/compute/buffer.h>
#include <vcl/compute/context.h>
#include <vcl/compute/kernel.h>
#include <vcl/compute/module.h>
#include <vcl/core/interleavedarray.h>

namespace Vcl { namespace Mathematics { namespace OpenCL
{
	class JacobiSVD33
	{
	public:
		JacobiSVD33(Core::ref_ptr<Compute::Context> ctx);

	public:
		void operator()
			(
			Vcl::Compute::CommandQueue& queue,
			const Vcl::Core::InterleavedArray<float, 3, 3, -1>& A,
			Vcl::Core::InterleavedArray<float, 3, 3, -1>& U,
			Vcl::Core::InterleavedArray<float, 3, 3, -1>& V,
			Vcl::Core::InterleavedArray<float, 3, 1, -1>& S
		);

	private:
		// Device context
		Core::ref_ptr<Compute::Context> _ownerCtx;

	private:
		// Module
		Core::ref_ptr<Compute::Module> _svdModule;

		// Kernel performing the SVD computation
		Core::ref_ptr<Compute::Kernel> _svdKernel;

	private: // Buffers
		
		//! Number of allocated entries
		size_t _capacity = 0;

		//! Input buffer 
		Core::ref_ptr<Compute::Buffer> _A;

		//! Output buffer 
		Core::ref_ptr<Compute::Buffer> _U;

		//! Output buffer 
		Core::ref_ptr<Compute::Buffer> _V;

		//! Singular value buffer 
		Core::ref_ptr<Compute::Buffer> _S;
	};
}}}
