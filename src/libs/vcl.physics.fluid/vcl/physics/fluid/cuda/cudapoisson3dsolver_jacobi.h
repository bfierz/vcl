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
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>

// VCL
#include <vcl/math/solver/jacobi.h>

namespace Vcl { namespace Mathematics { namespace Solver { class ConjugateGradients; }}}
namespace Vcl { namespace Physics { namespace Fluid { class CenterGrid; }}}
namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda { class CenterGrid; }}}}

#ifdef VCL_CUDA_SUPPORT

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/kernel.h>
#include <vcl/compute/cuda/module.h>

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	class CenterGrid3DPoissonJacobiCtx : public Vcl::Mathematics::Solver::JacobiContext
	{
	public:
		CenterGrid3DPoissonJacobiCtx
		(
			ref_ptr<Vcl::Compute::Cuda::Context> ctx,
			ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue,
			Eigen::Vector3i dim
		);
		virtual ~CenterGrid3DPoissonJacobiCtx();
		
	public:
		void setData
		(
			// Laplacian matrix (center, x, y, z)
			std::array<ref_ptr<Compute::Buffer>, 4> laplacian,

			// Blocked grid cells
			ref_ptr<Compute::Buffer> obstacles,

			// Pressure to keep the field divergence free
			ref_ptr<Compute::Buffer> pressure,

			// Divergence of the velocity field
			ref_ptr<Compute::Buffer> divergence,

			// Dimensions of the grid
			Eigen::Vector3i dim
		);

	public:
		virtual int size() const override;

		//! Resize the internal data structures of the context
		virtual void resize(int size) override;

	public:
		//
		virtual void precompute() override;

		// A x = b
		// -> A = D + R
		// -> x^{n+1} = D^-1 (b - R x^{n})
		virtual void updateSolution() override;

		//
		virtual double computeError()override;

		//! Ends the solver and returns the residual
		virtual void finish(double* residual) override;

	public: // Compute management methods
		Vcl::Compute::Context* context() { return _context.get(); }
		Vcl::Compute::CommandQueue* stream() { return _queue.get(); }

	private:
		ref_ptr<Vcl::Compute::Context> _context;
		ref_ptr<Vcl::Compute::CommandQueue> _queue;

	private: // Kernel functions

		//! Module with the cuda fluid 3d poisson jacobi kernel functions
		ref_ptr<Compute::Cuda::Module> _jacobiModule;

		//! Device function updating the solution
		ref_ptr<Compute::Cuda::Kernel> _jacobiUpdateSolution;

	private: // Associated grid
		//! Laplacian matrix (center, x, y, z)
		std::array<ref_ptr<Compute::Buffer>, 4> _laplacian;

		//! Blocked grid cells
		ref_ptr<Compute::Cuda::Buffer> _obstacles;

		//! Pressure to keep the field divergence free
		ref_ptr<Compute::Cuda::Buffer> _pressure;

		//! Divergence of the velocity field
		ref_ptr<Compute::Cuda::Buffer> _divergence;

		//! Temporary buffer for the updated solution
		ref_ptr<Compute::Cuda::Buffer> _solution;

		//! Dimensions of the grid
		Eigen::Vector3i _dim;
	};
}}}}
#endif /* VCL_CUDA_SUPPORT */
