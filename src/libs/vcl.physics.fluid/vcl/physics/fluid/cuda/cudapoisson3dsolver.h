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
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/module.h>
#include <vcl/math/cuda/conjugategradientscontext.h>
#include <vcl/physics/fluid/poisson3dsolver.h>

namespace Vcl { namespace Mathematics { namespace Solver { class ConjugateGradients; } } }
namespace Vcl { namespace Mathematics { namespace Solver { class Jacobi; } } }
namespace Vcl { namespace Mathematics { namespace Solver { namespace Cuda { class Poisson3DCgCtx; } } } }
namespace Vcl { namespace Mathematics { namespace Solver { namespace Cuda { class Poisson3DJacobiCtx; } } } }
namespace Vcl { namespace Physics { namespace Fluid { class CenterGrid; }}}
namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda { class CenterGrid; }}}}

#define VCL_PHYSICS_FLUID_CUDA_SOLVER_CG
//#define VCL_PHYSICS_FLUID_CUDA_SOLVER_JACOBI

#ifdef VCL_CUDA_SUPPORT
namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	class CenterGrid3DPoissonSolver : public Fluid::CenterGrid3DPoissonSolver
	{
	public:
		CenterGrid3DPoissonSolver
		(
			ref_ptr<Vcl::Compute::Cuda::Context> ctx,
			ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue
		);
		virtual ~CenterGrid3DPoissonSolver();

	public:
		virtual void updateSolver(Fluid::CenterGrid& g) override;
		virtual void makeDivergenceFree(Fluid::CenterGrid& g) override;
		virtual void diffuseField(Fluid::CenterGrid& g, float diffusion_constant) override;

	private: // Configurations

		//! Is the dynamic parallelism solver supported
		bool _supportsFullDeviceSolver = false;

		//! Is the separate device function solver supported
		bool _supportsPartialDeviceSolver = false;

	private: // Kernel functions

		//! Link to the context the solver was created with
		ref_ptr<Vcl::Compute::Cuda::Context> _ownerCtx;

		//! Associated command queue
		ref_ptr<Vcl::Compute::Cuda::CommandQueue> _queue;

		//! Module with the cuda fluid 3d poisson solver
		ref_ptr<Vcl::Compute::Cuda::Module> _module;

		//! Full device solver
		ref_ptr<Vcl::Compute::Cuda::Kernel> _fullDeviceSolver = nullptr;

		//! Device function computing the divergence of the velocity field
		ref_ptr<Vcl::Compute::Cuda::Kernel> _compDiv = nullptr;

		//! Device function building the lhs of the laplace matrix
		ref_ptr<Vcl::Compute::Cuda::Kernel> _buildLhs = nullptr;

		//! Correct the velocties with the computed pressures
		ref_ptr<Vcl::Compute::Cuda::Kernel> _correctField = nullptr;
		
	private: // Internal CG solver
#ifdef VCL_PHYSICS_FLUID_CUDA_SOLVER_CG
		//! Conjugate gradients solver used to solve the poisson equation
		std::unique_ptr<Vcl::Mathematics::Solver::ConjugateGradients> _solver;

		//! CUDA CG context to be used in conjunction with the solver
		std::unique_ptr<Vcl::Mathematics::Solver::Cuda::Poisson3DCgCtx> _solverCtx;
#endif

#ifdef VCL_PHYSICS_FLUID_CUDA_SOLVER_JACOBI
		//! Jacobi solver used to solve the poisson equation
		std::unique_ptr<Vcl::Mathematics::Solver::Jacobi> _solver;

		//! CUDA Jacobi context to be used in conjunction with the solver
		std::unique_ptr<Vcl::Mathematics::Solver::Cuda::Poisson3DJacobiCtx> _solverCtx;
#endif

	private: // Solver related memory

		//! Laplacian matrix (center, x, y, z)
		std::array<ref_ptr<Vcl::Compute::Cuda::Buffer>, 4> _laplacian;

		//! Pressure to keep the field divergence free
		ref_ptr<Vcl::Compute::Cuda::Buffer> _pressure;

		//! Divergence of the velocity field
		ref_ptr<Vcl::Compute::Cuda::Buffer> _divergence;
	};
}}}}
#endif // VCL_CUDA_SUPPORT
