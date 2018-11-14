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
#include <vcl/math/solver/cuda/poisson3dsolver_cg.h>
#include <vcl/math/solver/cuda/poisson3dsolver_jacobi.h>

// Tests
#include "poisson.h"

// Google test
#include <gtest/gtest.h>

TEST(Poisson3DCuda, MakeStencil)
{
	using namespace Vcl::Core;
	using namespace Vcl::Compute::Cuda;
	using namespace Vcl::Mathematics::Solver;

	using Vcl::Compute::BufferAccess;
	using Vcl::Compute::BufferView;

	Platform::initialise();
	Platform* platform = Platform::instance();
	const Device& device = platform->device(0);
	auto dev_ctx = Vcl::make_owner<Context>(device);
	dev_ctx->bind();

	auto queue = dev_ctx->createCommandQueue();

	std::vector<unsigned char> skip_cpu(10 * 10 * 10, 0);
	for (int z = 0; z < 10; z++)
	for (int y = 0; y < 10; y++)
	for (int x = 0; x < 10; x++)
	{
		int idx = 10 * 10 * z + 10 * y + x;
		if ((z > 0 && z < 4) || (z > 5 && z < 9) ||
			(y > 0 && y < 4) || (y > 5 && y < 9) ||
			(x > 0 && x < 4) || (x > 5 && x < 9))
			skip_cpu[idx] = 1;
	}
	auto skip_dev = static_pointer_cast<Buffer>(dev_ctx->createBuffer(BufferAccess::None, 10 * 10 * 10 * sizeof(unsigned char)));
	dev_ctx->defaultQueue()->write(BufferView{ skip_dev }, skip_cpu.data(), true);

	Cuda::Poisson3DJacobiCtx ctx_cpu{ dev_ctx, queue, { 10, 10, 10} };
	ctx_cpu.updatePoissonStencil(0.1f, -1, 0, { skip_cpu.data(), (int64_t)skip_cpu.size() });
	
	Cuda::Poisson3DJacobiCtx ctx_dev{ dev_ctx, queue, { 10, 10, 10} };
	ctx_dev.updatePoissonStencil(0.1f, -1, 0, *skip_dev);
	
	const auto matrix_cpu = ctx_cpu.matrix();
	const auto matrix_dev = ctx_dev.matrix();

	for (int i = 0; i < 7; i++)
	{
		std::vector<float> buf_cpu(10 * 10 * 10, 0);
		std::vector<float> buf_dev(10 * 10 * 10, 0);
		dev_ctx->defaultQueue()->read(buf_cpu.data(), matrix_cpu[i], false);
		dev_ctx->defaultQueue()->read(buf_dev.data(), matrix_dev[i], true);

		EXPECT_TRUE(std::equal(std::begin(buf_cpu), std::end(buf_cpu), std::begin(buf_dev), [](float a, float b) { return Vcl::Mathematics::equal(a, b); }));
	}

	dev_ctx->release(skip_dev);
}

TEST(Poisson3DCuda, SimpleJacobiNoBlockerIdentity)
{
	using namespace Vcl::Mathematics::Solver;

	float h;
	Eigen::VectorXf rhs, sol;
	std::vector<unsigned char> skip;
	unsigned int nr_pts = createPoisson3DProblem(h, rhs, sol, skip);

	Eigen::VectorXf lhs = sol;
	runPoissonTest<Jacobi, Cuda::Poisson3DJacobiCtx, Eigen::Vector3ui>({ nr_pts, nr_pts, nr_pts }, h, lhs, rhs, sol, skip, 1, 1e-4f);
}

TEST(Poisson3DCuda, SimpleJacobiNoBlocker)
{
	using namespace Vcl::Mathematics::Solver;

	float h;
	Eigen::VectorXf rhs, sol;
	std::vector<unsigned char> skip;
	unsigned int nr_pts = createPoisson3DProblem(h, rhs, sol, skip);

	Eigen::VectorXf lhs; lhs.setZero(nr_pts*nr_pts*nr_pts);
	runPoissonTest<Jacobi, Cuda::Poisson3DJacobiCtx, Eigen::Vector3ui>({ nr_pts, nr_pts, nr_pts }, h, lhs, rhs, sol, skip, 1000, 5e-3f);
}

TEST(Poisson3DCuda, SimpleCgNoBlockerIdentity)
{
	using namespace Vcl::Mathematics::Solver;

	float h;
	Eigen::VectorXf rhs, sol;
	std::vector<unsigned char> skip;
	unsigned int nr_pts = createPoisson3DProblem(h, rhs, sol, skip);

	Eigen::VectorXf lhs = sol;
	runPoissonTest<ConjugateGradients, Cuda::Poisson3DCgCtx, Eigen::Vector3ui>({ nr_pts, nr_pts, nr_pts }, h, lhs, rhs, sol, skip, 1, 1e-4f);
}

TEST(Poisson3DCuda, SimpleCgNoBlocker)
{
	using namespace Vcl::Mathematics::Solver;

	float h;
	Eigen::VectorXf rhs, sol;
	std::vector<unsigned char> skip;
	unsigned int nr_pts = createPoisson3DProblem(h, rhs, sol, skip);

	Eigen::VectorXf lhs; lhs.setZero(nr_pts*nr_pts*nr_pts);
	runPoissonTest<ConjugateGradients, Cuda::Poisson3DCgCtx, Eigen::Vector3ui>({ nr_pts, nr_pts, nr_pts }, h, lhs, rhs, sol, skip, 10, 1e-0f);
}
