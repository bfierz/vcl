/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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

// C++ standard library
#include <iostream>

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/math/math.h>
#include <vcl/math/rotation33_torque.h>

#include "problems.h"

// Common functions
namespace
{
	template<typename Scalar>
	void checkSolution
	(
		size_t nr_problems,
		Scalar tol,
		const std::vector<int>& iters,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refRa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resRa
	)
	{
		using Vcl::Mathematics::equal;

		Eigen::IOFormat fmt(6, 0, ", ", ";", "[", "]");

		for (int i = 0; i < static_cast<int>(nr_problems); i++)
		{
			Vcl::Matrix3f refR = refRa.template at<Scalar>(i);
			Vcl::Matrix3f resR = resRa.template at<Scalar>(i);

			Scalar sqLenRefRc0 = refR.col(0).squaredNorm();
			Scalar sqLenRefRc1 = refR.col(1).squaredNorm();
			Scalar sqLenRefRc2 = refR.col(2).squaredNorm();
			//EXPECT_TRUE(equal(sqLenRefRc0, Scalar(1), tol)) << "Reference R(" << i << "): Column 0 is not normalized.";
			//EXPECT_TRUE(equal(sqLenRefRc1, Scalar(1), tol)) << "Reference R(" << i << "): Column 1 is not normalized.";
			//EXPECT_TRUE(equal(sqLenRefRc2, Scalar(1), tol)) << "Reference R(" << i << "): Column 2 is not normalized.";

			Scalar sqLenResRc0 = resR.col(0).squaredNorm();
			Scalar sqLenResRc1 = resR.col(1).squaredNorm();
			Scalar sqLenResRc2 = resR.col(2).squaredNorm();
			//EXPECT_TRUE(equal(sqLenResRc0, Scalar(1), tol)) << "Result R(" << i << "): Column 0 is not normalized.";
			//EXPECT_TRUE(equal(sqLenResRc1, Scalar(1), tol)) << "Result R(" << i << "): Column 1 is not normalized.";
			//EXPECT_TRUE(equal(sqLenResRc2, Scalar(1), tol)) << "Result R(" << i << "): Column 2 is not normalized.";

			float ang_dist = Eigen::Quaternionf{ refR }.angularDistance(Eigen::Quaternionf{ resR });
			std::cout << "[" << ang_dist << ", " << iters[i] << "]" << std::endl;

			Vcl::Matrix3f I = resR.transpose() * resR;
			Vcl::Matrix3f Iref = Vcl::Matrix3f::Identity();

			//EXPECT_TRUE(equal(Iref, I, tol)) << "Index: " << i << ", Result U^t U: not Identity.";
		}
	}
}
template<typename WideScalar>
void runRotationTest(float max_angle, float tol)
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1>    F(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resR(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refR(nr_problems);
	
	createRotationProblems(nr_problems, max_angle * 3.14f / 180.0f, 0.7f, F, &refR);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	std::vector<int> iterations(nr_problems);
	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t A = F.at<real_t>(i);
		matrix3_t R = A;

		iterations[i] = Vcl::Mathematics::Rotation(A, R);


		resR.at<real_t>(i) = R;
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, iterations, refR, resR);
}

int main(int, char**)
{
	runRotationTest<float>(90.0f, 2e-5f);
	return 0;
}
