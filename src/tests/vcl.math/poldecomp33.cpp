/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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

// Include the relevant parts from the library
#include <vcl/core/interleavedarray.h>
#include <vcl/math/math.h>
#include <vcl/math/polardecomposition.h>

// Google test
#include <gtest/gtest.h>

// Common functions
template<typename Scalar>
Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> createProblems(size_t nr_problems)
{
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> F(nr_problems);

	// Initialize data
	for (int i = 0; i < (int) nr_problems; i++)
	{
		F.at<Scalar>(i).setRandom();
	}

	return std::move(F);
}

template<typename Scalar>
void computeReferenceSolution
(
	size_t nr_problems,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& R,
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& S
)
{
	// Compute reference using Eigen
	for (int i = 0; i < static_cast<int>(nr_problems); i++)
	{
		Vcl::Matrix3f A = F.at<Scalar>(i);

		Eigen::JacobiSVD<Eigen::Matrix<Scalar, 3, 3>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

		// Adapted the polar decomposition from Eigen
		Scalar x = (svd.matrixU() * svd.matrixV().adjoint()).determinant();
		Eigen::Matrix<Scalar, 3, 1> sv(svd.singularValues());

		int index;
		sv.minCoeff(&index);

		Eigen::Matrix<Scalar, 3, 3> V(svd.matrixV());
		V.col(index) /= x;
		sv.coeffRef(index) /= x;

		R.at<Scalar>(i) = svd.matrixU() * V.adjoint();
		S.at<Scalar>(i) = V * sv.asDiagonal() * V.adjoint();
	}
}

template<typename Scalar>
void checkSolution
(
	size_t nr_problems,
	Scalar tol,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refRa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refSa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resRa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resSa
)
{
	using Vcl::Mathematics::equal;

	for (int i = 0; i < static_cast<int>(nr_problems); i++)
	{
		Vcl::Matrix3f refR = refRa.at<Scalar>(i);
		Vcl::Matrix3f refS = refSa.at<Scalar>(i);

		Vcl::Matrix3f resR = resRa.at<Scalar>(i);
		Vcl::Matrix3f resS = resSa.at<Scalar>(i);

		Scalar sqLenRefRc0 = refR.col(0).squaredNorm();
		Scalar sqLenRefRc1 = refR.col(1).squaredNorm();
		Scalar sqLenRefRc2 = refR.col(2).squaredNorm();
		EXPECT_TRUE(equal(sqLenRefRc0, Scalar(1), tol)) << "Reference R(" << i << "): Column 0 is not normalized.";
		EXPECT_TRUE(equal(sqLenRefRc1, Scalar(1), tol)) << "Reference R(" << i << "): Column 1 is not normalized.";
		EXPECT_TRUE(equal(sqLenRefRc2, Scalar(1), tol)) << "Reference R(" << i << "): Column 2 is not normalized.";

		Scalar sqLenResRc0 = resR.col(0).squaredNorm();
		Scalar sqLenResRc1 = resR.col(1).squaredNorm();
		Scalar sqLenResRc2 = resR.col(2).squaredNorm();
		EXPECT_TRUE(equal(sqLenResRc0, Scalar(1), tol)) << "Result R(" << i << "): Column 0 is not normalized.";
		EXPECT_TRUE(equal(sqLenResRc1, Scalar(1), tol)) << "Result R(" << i << "): Column 1 is not normalized.";
		EXPECT_TRUE(equal(sqLenResRc2, Scalar(1), tol)) << "Result R(" << i << "): Column 2 is not normalized.";

		bool eqS = refS.array().abs().isApprox(resS.array().abs(), tol);
		bool eqR = refR.array().abs().isApprox(resR.array().abs(), tol);

		EXPECT_TRUE(eqS) << "S(" << i << ") - Ref: " << refS << ", Actual: " << resS;
		EXPECT_TRUE(eqR) << "R(" << i << ") - Ref: " << refR << ", Actual: " << resR;
	}
}

template<typename WideScalar>
void runPolDecompTest(float tol)
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resR(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refR(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refS(nr_problems);

	auto F = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, F, refR, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t A = F.at<real_t>(i);
		matrix3_t R, S;

		Vcl::Mathematics::PolarDecomposition(A, R, &S);

		resR.at<real_t>(i) = R;
		resS.at<real_t>(i) = S;
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refR, refS, resR, resS);
}

TEST(PolarDecomposition33, PolDecompFloat)
{
	runPolDecompTest<float>(1e-5f);
}
TEST(PolarDecomposition33, PolDecompFloat4)
{
	runPolDecompTest<Vcl::float4>(1e-5f);
}
TEST(PolarDecomposition33, PolDecompFloat8)
{
	runPolDecompTest<Vcl::float8>(1e-5f);
}
