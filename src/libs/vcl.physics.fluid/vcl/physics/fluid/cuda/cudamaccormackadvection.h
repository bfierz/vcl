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
#include <vcl/config/eigen.h>
#include <vcl/config/cuda.h>

// C++ standard library

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/module.h>
#include <vcl/physics/fluid/cuda/cudasemilagrangeadvection.h>

#ifdef VCL_CUDA_SUPPORT
namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	class MacCormackAdvection : public Advection
	{
	public:
		MacCormackAdvection(ref_ptr<Compute::Cuda::Context> ctx);
		MacCormackAdvection(ref_ptr<Compute::Cuda::Context> ctx, unsigned int x, unsigned int y, unsigned int z);
		virtual ~MacCormackAdvection();

		virtual void setSize(unsigned int x, unsigned int y, unsigned int z) override;
		
	public:
		virtual void operator()
		(
			ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue,
			float dt,
			const Fluid::CenterGrid* grid,
			ref_ptr<const Compute::Buffer> src,
			ref_ptr<Compute::Buffer> dst
		) override;

	private:

		//! Device function performing a semi-lagrange advection step
		ref_ptr<Compute::Cuda::Kernel> _advect{ nullptr };
		
		//! Device function merging the forward and backward advection
		ref_ptr<Compute::Cuda::Kernel> _merge{ nullptr };

		//! Device function clamping the advection results
		ref_ptr<Compute::Cuda::Kernel> _clamp{ nullptr };

	private:
		//! Intermediate buffer
		ref_ptr<Compute::Cuda::Buffer> _intermediateBuffer;
	};
}}}}
#endif // VCL_CUDA_SUPPORT
