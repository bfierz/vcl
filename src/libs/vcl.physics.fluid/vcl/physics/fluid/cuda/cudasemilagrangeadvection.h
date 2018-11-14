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
#include <vcl/physics/fluid/advection.h>

//#define VCL_FLUID_ADVECT_TEX3D

#ifdef VCL_CUDA_SUPPORT
namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	class Advection
	{
	public:
		Advection(ref_ptr<Compute::Cuda::Context> ctx);
		Advection(ref_ptr<Compute::Cuda::Context> ctx, unsigned int x, unsigned int y, unsigned int z);
		virtual ~Advection() = default;
		
	public:
		virtual void operator()
		(
			ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue,
			float dt,
			const Fluid::CenterGrid* grid,
			ref_ptr<const Compute::Buffer> src,
			ref_ptr<Compute::Buffer> dst
		) =0;

	public:
		unsigned int x() const { return _x; }
		unsigned int y() const { return _y; }
		unsigned int z() const { return _z; }

	public:
		virtual void setSize(unsigned int x, unsigned int y, unsigned int z)
		{
			_x = x;
			_y = y;
			_z = z;
		}

	protected:
		ref_ptr<Compute::Cuda::Context> context() { return _ownerCtx; }

	private:
		//! Link to the context the solver was created with
		ref_ptr<Compute::Cuda::Context> _ownerCtx{ nullptr };

	protected:
		//! Module with the cuda advection code
		ref_ptr<Compute::Cuda::Module> _advectModule;

	private:
		unsigned int _x; //!< Grid resolution - x
		unsigned int _y; //!< Grid resolution - y
		unsigned int _z; //!< Grid resolution - z
	};

	class SemiLagrangeAdvection : public Advection
	{
	public:
		SemiLagrangeAdvection(ref_ptr<Compute::Cuda::Context> ctx);
		SemiLagrangeAdvection(ref_ptr<Compute::Cuda::Context> ctx, unsigned int x, unsigned int y, unsigned int z);
		virtual ~SemiLagrangeAdvection();

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
		ref_ptr<Compute::Cuda::Kernel> _advect = nullptr;
		
		//! Texture based device function performing the advection step
		ref_ptr<Compute::Cuda::Kernel> _advectTex = nullptr;

#ifdef VCL_FLUID_ADVECT_TEX3D
		ref_ptr<Compute::Cuda::TextureReference> mTexOldField;

	private:
		CUarray mDevArrOldField;
#endif // VCL_FLUID_ADVECT_TEX3D
	};
}}}}
#endif // VCL_CUDA_SUPPORT
