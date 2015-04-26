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
#include <vcl/math/jacobisvd33_mcadams.h>
#include <vcl/math/jacobisvd33_qr.h>
#include <vcl/math/jacobisvd33_twosided.h>

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
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& U,
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& V,
	Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& S
)
{
	// Compute reference using Eigen
	for (int i = 0; i < static_cast<int>(nr_problems); i++)
	{
		Vcl::Matrix3f A = F.at<Scalar>(i);
		Eigen::JacobiSVD<Vcl::Matrix3f> eigen_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
		U.at<Scalar>(i) = eigen_svd.matrixU();
		V.at<Scalar>(i) = eigen_svd.matrixV();
		S.at<Scalar>(i) = eigen_svd.singularValues();
	}
}

template<typename Scalar>
void checkSolution
(
	size_t nr_problems,
	Scalar tol,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refUa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refVa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& refSa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resUa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resVa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& resSa
)
{
	for (int i = 0; i < static_cast<int>(nr_problems); i++)
	{
		Vcl::Matrix3f refU = refUa.at<Scalar>(i);
		Vcl::Matrix3f refV = refVa.at<Scalar>(i);
		Vcl::Vector3f refS = refSa.at<Scalar>(i);

		Vcl::Matrix3f resU = resUa.at<Scalar>(i);
		Vcl::Matrix3f resV = resVa.at<Scalar>(i);
		Vcl::Vector3f resS = resSa.at<Scalar>(i);

		bool eqU = refU.array().abs().isApprox(resU.array().abs(), tol);
		bool eqV = refV.array().abs().isApprox(resV.array().abs(), tol);
		bool eqS = refS.array().abs().isApprox(resS.array().abs(), tol);

		EXPECT_TRUE(eqU) << "U(" << i << ")";
		EXPECT_TRUE(eqV) << "V(" << i << ")";
		EXPECT_TRUE(eqS) << "S(" << i << ")";
	}
}

template<typename WideScalar>
void runMcAdamsTest()
{
	using scalar_t  = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	// Checking tolerance
	scalar_t tol = scalar_t(1e-5);

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	auto F = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, F, refU, refV, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t S = F.at<real_t>(i);
		matrix3_t U = matrix3_t::Identity();
		matrix3_t V = matrix3_t::Identity();

		Vcl::Mathematics::McAdamsJacobiSVD(S, U, V);
		
		resU.at<real_t>(i) = U;
		resV.at<real_t>(i) = V;
		resS.at<real_t>(i) = S.diagonal();
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refV, refS, resU, resV, resS);
}

template<typename WideScalar>
void runTwoSidedTest()
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	// Checking tolerance
	scalar_t tol = scalar_t(1e-5);

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	auto F = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, F, refU, refV, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t S = F.at<real_t>(i);
		matrix3_t U = matrix3_t::Identity();
		matrix3_t V = matrix3_t::Identity();

		Vcl::Mathematics::TwoSidedJacobiSVD(S, U, V);

		resU.at<real_t>(i) = U;
		resV.at<real_t>(i) = V;
		resS.at<real_t>(i) = S.diagonal();
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refV, refS, resU, resV, resS);
}

template<typename WideScalar>
void runQRTest()
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	// Checking tolerance
	scalar_t tol = scalar_t(1e-5);

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	auto F = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, F, refU, refV, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t S = F.at<real_t>(i);
		matrix3_t U = matrix3_t::Identity();
		matrix3_t V = matrix3_t::Identity();

		Vcl::Mathematics::QRJacobiSVD(S, U, V);

		resU.at<real_t>(i) = U;
		resV.at<real_t>(i) = V;
		resS.at<real_t>(i) = S.diagonal();
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refV, refS, resU, resV, resS);
}

TEST(SVD33, McAdamsSVDFloat)
{
	runMcAdamsTest<float>();
}
TEST(SVD33, McAdamsSVDFloat4)
{
	runMcAdamsTest<Vcl::float4>();
}
TEST(SVD33, McAdamsSVDFloat8)
{
	runMcAdamsTest<Vcl::float8>();
}

TEST(SVD33, TwoSidedSVDFloat)
{
	runTwoSidedTest<float>();
}
TEST(SVD33, TwoSidedSVDFloat4)
{
	runTwoSidedTest<Vcl::float4>();
}
TEST(SVD33, TwoSidedSVDFloat8)
{
	runTwoSidedTest<Vcl::float8>();
}

TEST(SVD33, QRSVDFloat)
{
	runQRTest<float>();
}
TEST(SVD33, QRSVDFloat4)
{
	runQRTest<Vcl::float4>();
}
TEST(SVD33, QRSVDFloat8)
{
	runQRTest<Vcl::float8>();
}
