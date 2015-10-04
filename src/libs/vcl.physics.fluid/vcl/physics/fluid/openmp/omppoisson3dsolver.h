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

// C++ standard library
#include <array>
#include <memory>

// OpenMP library
#include <omp.h>

// VCL
#include <vcl/mathematics/solver/eigenconjugategradientscontext.h>
#include <vcl/physics/fluid/poisson3dsolver.h>

namespace Vcl { namespace Mathematics { namespace Solver { class ConjugateGradients; }}}
namespace Vcl { namespace Physics { namespace Fluid { class CenterGrid; }}}

#ifdef VCL_OPENMP_SUPPORT
namespace Vcl { namespace Physics { namespace Fluid { namespace OpenMP
{
	class CenterGrid3DPoissonCgCtx : public Vcl::Mathematics::Solver::EigenCgBaseContext<float>
	{
	public:
		CenterGrid3DPoissonCgCtx(Eigen::Vector3i dim);
		virtual ~CenterGrid3DPoissonCgCtx();
		
	public:
		void setData
		(
			std::array<Eigen::VectorXf, 4>* laplacian,
			Eigen::Map<Eigen::VectorXi>* obstacles,
			Eigen::Map<Eigen::VectorXf>* pressure,
			Eigen::Map<Eigen::VectorXf>* divergence,
			Eigen::Vector3i dim
		);

	public:
		// d = r = b - A*x
		virtual void computeInitialResidual() override;

		// q = A*d
		virtual void computeQ() override;

	private: // Associated grid
		//! Laplacian matrix (center, x, y, z)
		std::array<Eigen::VectorXf, 4>* _laplacian;

		//! Blocked grid cells
		Eigen::Map<Eigen::VectorXi>* _obstacles;

		//! Pressure to keep the field divergence free
		Eigen::Map<Eigen::VectorXf>* _pressure;

		//! Divergence of the velocity field
		Eigen::Map<Eigen::VectorXf>* _divergence;

		//! Dimensions of the grid
		Eigen::Vector3i _dim;
	};

	class CenterGrid3DPoissonSolver : public Fluid::CenterGrid3DPoissonSolver
	{
	public:
		CenterGrid3DPoissonSolver();
		virtual ~CenterGrid3DPoissonSolver();

	public:
		virtual void solve(Fluid::CenterGrid& g) override;

	private: /* Solving steps */
		void computeDivergence
		(
			Eigen::Map<Eigen::VectorXf>* divergence,
			Eigen::Map<Eigen::VectorXf>* pressure,
			Eigen::Map<Eigen::VectorXf>* vel_x,
			Eigen::Map<Eigen::VectorXf>* vel_y,
			Eigen::Map<Eigen::VectorXf>* vel_z,
			const Eigen::VectorXi& obstacles,
			Eigen::Vector3i  dim,
			float cell_size
		);

		void buildLHS
		(
			Eigen::VectorXf& Acenter,
			Eigen::VectorXf& Ax,
			Eigen::VectorXf& Ay,
			Eigen::VectorXf& Az,
			const Eigen::VectorXi& obstacles,
			Eigen::Vector3i dim
		);

	private: // Internal CG solver

		//! Conjugate gradients solver used to solve the poisson equation
		std::unique_ptr<Vcl::Mathematics::Solver::ConjugateGradients> _solver;

		//! CUDA CG context to be used in conjunction with the solver
		std::unique_ptr<CenterGrid3DPoissonCgCtx> _solverCtx;		

	private: // Solver related memory

		//! Laplacian matrix (center, x, y, z)
		std::array<Eigen::VectorXf, 4> _laplacian;

		//! Pressure to keep the field divergence free
		Eigen::VectorXf _pressure;

		//! Divergence of the velocity field
		Eigen::VectorXf _divergence;
	};
}}}}
#endif /* VCL_OPENMP_SUPPORT */
