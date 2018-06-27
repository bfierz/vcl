/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2016 Basil Fierz
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

// VCL
#include <vcl/math/solver/jacobi.h>
#include <vcl/math/math.h>

// Google test
#include <gtest/gtest.h>

inline Eigen::MatrixXf createStencilMatrix1D(unsigned int size, std::vector<unsigned char>& boundary)
{
	Eigen::MatrixXf stencil;
	stencil.setZero(size, size);
	for (Eigen::MatrixXf::Index i = 0; i < static_cast<Eigen::MatrixXf::Index>(size); i++)
	{
		if (!boundary[i])
		{
			if (i < (size - 1))
			{
				stencil(i, i) -= 1;
				stencil(i, i + 1) = 1;
			}
			if (i > 0)
			{
				stencil(i, i) -= 1;
				stencil(i, i - 1) = 1;
			}
		}
		else
		{
			stencil(i, i) = 1;
		}
	}

	return stencil;
}

class GenericJacobiCtx : public Vcl::Mathematics::Solver::JacobiContext
{
	using real_t = float;
	using matrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;
	using vector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;
	using map_t = Eigen::Map<vector_t>;

public:
	GenericJacobiCtx(unsigned int size)
	: _size(size)
	{
		_next.setZero(size);
	}

	void setData(gsl::not_null<map_t*> unknowns, gsl::not_null<map_t*> rhs)
	{
		_unknowns = unknowns;
		_rhs = rhs;
	}

	void updatePoissonStencil(real_t h, real_t k, Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip)
	{
		std::vector<unsigned char> boundary(skip.data(), skip.data() + skip.size());

		const real_t s = k / (h*h);
		_M = s*createStencilMatrix1D(_size, boundary);

		_D = (1.0f / (_M.diagonal()).array()).matrix().asDiagonal();
		for (int i = 0; i < skip.size(); i++)
		{
			if (skip[i] != 0)
				_D(i, i) = 0;
		}

		_R = _M;
		_R.diagonal().setZero();
	}

	int size() const override
	{
		return _size;
	}

	void precompute() override
	{
		_error = 0;
	}

	// A x = b
	// -> A = D + R
	// -> x^{n+1} = D^-1 (b - R x^{n})
	void updateSolution() override
	{
		auto& unknowns = *_unknowns;
		auto& rhs = *_rhs;

		// x^{n+1} = D^-1 (b - R x^{n})
		_next = _D * (rhs - _R * unknowns);
		for (int i = 0; i < unknowns.size(); i++)
		{
			if (_D(i, i) != 0)
				unknowns(i) = _next(i);
		}

		auto e = rhs - _M * unknowns;
		_error = e.squaredNorm();
	}

	//
	double computeError() override
	{
		return sqrt(_error) / size();
	}

	//! Ends the solver and returns the residual
	void finish(double*) override {}

private:
	//! Dimensions of the grid
	unsigned int _size;

	//! Current error
	float _error{ 0 };

	//! Laplacian matrix
	matrix_t _M;
	matrix_t _D;
	matrix_t _R;

	//! Left-hand side 
	map_t* _unknowns;

	//! Right-hand side
	map_t* _rhs;

	//! Temporary buffer for the updated solution
	vector_t _next;
};

// Example with exact solution taken from
// Burkardt - Jacobi Iterative Solution of Poisson's Equation in 1D
inline unsigned int createPoisson1DProblem(float& h, Eigen::VectorXf& rhs, Eigen::VectorXf& sol, std::vector<unsigned char>& boundary)
{
	// Number of points
	unsigned int nr_pts = 33;

	// Domain [0, 1]
	h = 1.0f / static_cast<float>(nr_pts - 1);

	// Right-hand side and exact solution of the poisson problem
	rhs.setZero(nr_pts);
	sol.setZero(nr_pts);
	for (Eigen::VectorXf::Index i = 0; i < static_cast<Eigen::VectorXf::Index>(nr_pts); i++)
	{
		float x = i * h;
		rhs(i) = -x * (x + 3) * exp(x);
		sol(i) =  x * (x - 1) * exp(x);
	}

	// Configure boundary condition
	boundary.assign(nr_pts, 0);
	boundary[0] = 1;
	boundary[nr_pts - 1] = 1;

	rhs(0) = 0;
	sol(0) = 0;
	rhs(nr_pts - 1) = 0;
	sol(nr_pts - 1) = 0;

	return nr_pts;
}

inline unsigned int createPoisson2DProblem(float& h, Eigen::VectorXf& rhs, Eigen::VectorXf& sol, std::vector<unsigned char>& boundary)
{
	using Vcl::Mathematics::pi;

	// Number of points
	unsigned int nr_pts = 16;

	// Domain [0, 1] x [0, 1]
	h = 1.0f / static_cast<float>(nr_pts - 1);

	// Right-hand side and exact solution of the poisson problem
	rhs.setZero(nr_pts*nr_pts);
	sol.setZero(nr_pts*nr_pts);
	boundary.assign(nr_pts*nr_pts, 0);
	for (Eigen::VectorXf::Index j = 0; j < static_cast<Eigen::VectorXf::Index>(nr_pts); j++)
	{
		for (Eigen::VectorXf::Index i = 0; i < static_cast<Eigen::VectorXf::Index>(nr_pts); i++)
		{
			float x = 0 + i * h;
			float y = 0 + j * h;
			sol(j*nr_pts + i) = sin(pi<float>()*x)*sin(pi<float>()*y);
			rhs(j*nr_pts + i) = 2*pi<float>()*pi<float>()*sin(pi<float>()*x)*sin(pi<float>()*y);

			if (i == 0 || i == nr_pts - 1 || j == 0 || j == nr_pts - 1)
				boundary[j*nr_pts + i] = 1;
		}
	}

	return nr_pts;
}

inline unsigned int createPoisson3DProblem(float& h, Eigen::VectorXf& rhs, Eigen::VectorXf& sol, std::vector<unsigned char>& boundary)
{
	using Vcl::Mathematics::pi;

	// Number of points
	unsigned int nr_pts = 8;

	// Domain [0, 1] x [0, 1] x [0, 1]
	h = 1.0f / static_cast<float>(nr_pts - 1);

	// Right-hand side and exact solution of the poisson problem
	rhs.setZero(nr_pts*nr_pts*nr_pts);
	sol.setZero(nr_pts*nr_pts*nr_pts);
	boundary.assign(nr_pts*nr_pts*nr_pts, 0);
	for (Eigen::VectorXf::Index k = 0; k < static_cast<Eigen::VectorXf::Index>(nr_pts); k++)
	{
		for (Eigen::VectorXf::Index j = 0; j < static_cast<Eigen::VectorXf::Index>(nr_pts); j++)
		{
			for (Eigen::VectorXf::Index i = 0; i < static_cast<Eigen::VectorXf::Index>(nr_pts); i++)
			{
				float x = 0 + i * h;
				float y = 0 + j * h;
				float z = 0 + k * h;
				sol(k*nr_pts*nr_pts + j*nr_pts + i) = sin(pi<float>()*x)*sin(pi<float>()*y)*sin(pi<float>()*z);
				rhs(k*nr_pts*nr_pts + j*nr_pts + i) = 3*pi<float>()*pi<float>()*sin(pi<float>()*x)*sin(pi<float>()*y)*sin(pi<float>()*z);

				if (i == 0 || i == nr_pts - 1 || j == 0 || j == nr_pts - 1 || k == 0 || k == nr_pts - 1)
					boundary[k*nr_pts*nr_pts + j*nr_pts + i] = 1;
			}
		}
	}

	return nr_pts;
}

template<typename Solver, typename Ctx, typename Dim>
void runPoissonTest(Dim dim, float h, Eigen::VectorXf& lhs, Eigen::VectorXf& rhs, const Eigen::VectorXf& sol, const std::vector<unsigned char>& skip, int max_iters, float eps)
{
	Eigen::Map<Eigen::VectorXf> x(lhs.data(), lhs.size());
	Eigen::Map<Eigen::VectorXf> y(rhs.data(), rhs.size());
	Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> boundary(skip.data(), (int64_t)skip.size());

	Ctx ctx{ dim };
	ctx.updatePoissonStencil(h, -1, boundary);
	ctx.setData(&x, &y);

	// Execute the poisson solver
	Solver solver;
	solver.setMaxIterations(max_iters);
	solver.solve(&ctx);

	// Check for the solution
	Eigen::VectorXf err; err.setZero(sol.size());
	for (Eigen::VectorXf::Index i = 0; i < sol.size(); i++)
	{
		err(i) = fabs(lhs(i) - sol(i));
	}

	Eigen::VectorXf::Index max_err_idx;
	EXPECT_LE(err.maxCoeff(&max_err_idx), eps) << "Maximum error at index " << max_err_idx;
}