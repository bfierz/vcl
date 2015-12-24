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
#include <array>

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/cuda/module.h>
#include <vcl/math/solver/conjugategradients.h>

#ifdef VCL_CUDA_SUPPORT
namespace Vcl { namespace Mathematics { namespace Solver { namespace Cuda
{
	class ConjugateGradientsContext : public Vcl::Mathematics::Solver::ConjugateGradientsContext
	{
	public:
		ConjugateGradientsContext(ref_ptr<Compute::Context> ctx, ref_ptr<Compute::CommandQueue> queue, int size);
		virtual ~ConjugateGradientsContext();

	public:
		virtual int size() const override;

	public:

		virtual void resize(int size) override;

		virtual void setX(ref_ptr<Compute::Buffer> x);

	public:
		// d = r = b - A*x
		virtual void computeInitialResidual() =0;
			
		// q = A*d
		virtual void computeQ() =0 ;
			
		// d_r = dot(r, r)
		// d_g = dot(d, q)
		// d_b = dot(r, q)
		// d_a = dot(q, q)
		virtual void reduceVectors() override;

		// alpha = d_r / d_g;
		// beta = d_r - 2.0f * alpha * d_b + alpha * alpha * d_a;
		// x = x + alpha * d
		// r = r - alpha * q
		// d = r + beta * d
		virtual void updateVectors() override;

		// abs(beta * d_r);
		virtual double computeError() override;

		//! Called after the last iteration. Returns d_r
		virtual void finish(double* residual = nullptr) override;

	protected:
		ref_ptr<Compute::Context> context() { return _ownerCtx; }

	private:
		void init();
		void destroy();

	private:
		// Device context
		ref_ptr<Compute::Context> _ownerCtx;

		//! Commandqueue the execution uses
		ref_ptr<Compute::CommandQueue> _queue;

	private:
		// Module
		ref_ptr<Compute::Module> _reduceUpdateModule;

		// Kernel performing the dot-product
		ref_ptr<Compute::Cuda::Kernel> _reduceBeginKernel;
		ref_ptr<Compute::Cuda::Kernel> _reduceContinueKernel;
		ref_ptr<Compute::Cuda::Kernel> _updateKernel;
		
	private: // Buffers for reduction
		std::array<ref_ptr<Compute::Cuda::Buffer>, 2> _reduceBuffersR;
		std::array<ref_ptr<Compute::Cuda::Buffer>, 2> _reduceBuffersG;
		std::array<ref_ptr<Compute::Cuda::Buffer>, 2> _reduceBuffersB;
		std::array<ref_ptr<Compute::Cuda::Buffer>, 2> _reduceBuffersA;
		
		float* _hostR;
		float* _hostG;
		float* _hostB;
		float* _hostA;

	private: /* Problem configuration */
		size_t _size;

	protected:
		ref_ptr<Compute::Cuda::Buffer> _devX;
		ref_ptr<Compute::Cuda::Buffer> _devDirection;
		ref_ptr<Compute::Cuda::Buffer> _devQ;
		ref_ptr<Compute::Cuda::Buffer> _devResidual;

	protected:
		//! Residual to reduce for the CG solver
		ref_ptr<Compute::Cuda::Buffer> _cgResidual;

		//! Update direction for the CG solver
		ref_ptr<Compute::Cuda::Buffer> _cgDirection;
	};
}}}}
#endif // VCL_CUDA_SUPPORT
