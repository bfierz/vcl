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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <random>

// Include the relevant parts from the library
#include <vcl/math/solver/poisson3dsolver_cg.h>
#include <vcl/math/solver/poisson3dsolver_jacobi.h>

// Tests
#include "poisson.h"

// Google test
#include <gtest/gtest.h>

TEST(Poisson3D, SimpleJacobiNoBlockerIdentity)
{
	using namespace Vcl::Mathematics::Solver;

	float h;
	Eigen::VectorXf rhs, sol;
	unsigned int nr_pts = createPoisson3DProblem(h, rhs, sol);

	runPoissonTest<Jacobi, Poisson3DJacobiCtx<float>, Eigen::Vector3ui>({ nr_pts, nr_pts, nr_pts }, h, sol, rhs, sol, 1, 1e-5f);
}

TEST(Poisson3D, SimpleJacobiNoBlocker)
{
	using namespace Vcl::Mathematics::Solver;

	float h;
	Eigen::VectorXf rhs, sol;
	unsigned int nr_pts = createPoisson3DProblem(h, rhs, sol);

	Eigen::VectorXf lhs; lhs.setZero(nr_pts*nr_pts*nr_pts);
	runPoissonTest<Jacobi, Poisson3DJacobiCtx<float>, Eigen::Vector3ui>({ nr_pts, nr_pts, nr_pts }, h, lhs, rhs, sol, 100, 1e+1f);
}

TEST(Poisson3D, SimpleCgNoBlockerIdentity)
{
	using namespace Vcl::Mathematics::Solver;

	float h;
	Eigen::VectorXf rhs, sol;
	unsigned int nr_pts = createPoisson3DProblem(h, rhs, sol);

	runPoissonTest<ConjugateGradients, Poisson3DCgCtx<float>, Eigen::Vector3ui>({ nr_pts, nr_pts, nr_pts }, h, sol, rhs, sol, 1, 1e-5f);
}

TEST(Poisson3D, SimpleCgNoBlocker)
{
	using namespace Vcl::Mathematics::Solver;

	float h;
	Eigen::VectorXf rhs, sol;
	unsigned int nr_pts = createPoisson3DProblem(h, rhs, sol);

	Eigen::VectorXf lhs = rhs;
	runPoissonTest<ConjugateGradients, Poisson3DCgCtx<float>, Eigen::Vector3ui>({ nr_pts, nr_pts, nr_pts }, h, lhs, rhs, sol, nr_pts*nr_pts*nr_pts, 1e+1f);
}
