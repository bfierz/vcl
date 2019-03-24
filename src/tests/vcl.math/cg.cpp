/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <vcl/math/solver/conjugategradients.h>
#include <vcl/math/solver/eigenconjugategradientscontext.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(ConjugateGradients, Identity)
{
	using namespace Vcl::Mathematics::Solver;

	Eigen::MatrixXf A = Eigen::MatrixXf::Identity(10, 10);
	Eigen::VectorXf b = Eigen::VectorXf::Ones(10);
	Eigen::VectorXf x = Eigen::VectorXf::Zero(10);
	for (int i = 0; i < 10; i++)
		b[i] = static_cast<float>(i + 1);

	GenericEigenCgContext<Eigen::MatrixXf> ctx{&A, &b};

	Eigen::Map<Eigen::VectorXf> mx{x.data(), x.size()};
	ctx.setX(mx);

	ConjugateGradients solver;
	solver.setMaxIterations(10);

	double residual = 0;
	solver.solve(&ctx, &residual);

	EXPECT_DOUBLE_EQ(residual, 0.0);
}
