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

// C++ standard library
#include <random>

// Include the relevant parts from the library
#include <vcl/core/interleavedarray.h>
#include <vcl/math/math.h>
#include <vcl/math/jacobisvd33_mcadams.h>
#include <vcl/math/jacobisvd33_qr.h>
#include <vcl/math/jacobisvd33_twosided.h>

// Google test
#include <gtest/gtest.h>

// Common functions
namespace
{
	template<typename Scalar>
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> createProblems(size_t nr_problems)
	{
		// Random number generator
		std::mt19937_64 rng{ 5489 };
		std::uniform_real_distribution<float> d;

		Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> F(nr_problems);
	
		// Initialize data
		for (int i = 0; i < (int) nr_problems; i++)
		{
			if (i < 8)
			{
				F.template at<Scalar>(i) = Eigen::Matrix<Scalar, 3, 3>::Identity();
			}
			else if (i < 16)
			{
				F.template at<Scalar>(i) = 0.35f * Eigen::Matrix<Scalar, 3, 3>::Identity();
			}
			else
			{
				Eigen::Matrix<Scalar, 3, 3> rnd;
				rnd << d(rng), d(rng), d(rng),
					   d(rng), d(rng), d(rng),
					   d(rng), d(rng), d(rng);
				F.template at<Scalar>(i) = rnd;
			}
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
			Vcl::Matrix3f A = F.template at<Scalar>(i);
			Eigen::JacobiSVD<Vcl::Matrix3f> eigen_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
			U.template at<Scalar>(i) = eigen_svd.matrixU();
			V.template at<Scalar>(i) = eigen_svd.matrixV();
			S.template at<Scalar>(i) = eigen_svd.singularValues();
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
		using Vcl::Mathematics::equal;

		Eigen::IOFormat fmt(6, 0, ", ", ";", "[", "]");

		for (int i = 0; i < static_cast<int>(nr_problems); i++)
		{
			Vcl::Matrix3f refU = refUa.template at<Scalar>(i);
			Vcl::Matrix3f refV = refVa.template at<Scalar>(i);
			Vcl::Vector3f refS = refSa.template at<Scalar>(i);

			Vcl::Matrix3f resU = resUa.template at<Scalar>(i);
			Vcl::Matrix3f resV = resVa.template at<Scalar>(i);
			Vcl::Vector3f resS = resSa.template at<Scalar>(i);

			if (refS(0) > 0 && refS(1) > 0 && refS(2) < 0)
				refV.col(2) *= -1;
			if (resS(0) > 0 && resS(1) > 0 && resS(2) < 0)
				resV.col(2) *= -1;

			Vcl::Matrix3f refR = refU * refV.transpose();
			Vcl::Matrix3f resR = resU * resV.transpose();

			Scalar sqLenRefUc0 = refU.col(0).squaredNorm();
			Scalar sqLenRefUc1 = refU.col(1).squaredNorm();
			Scalar sqLenRefUc2 = refU.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenRefUc0, Scalar(1), tol)) << "Reference U(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefUc1, Scalar(1), tol)) << "Reference U(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefUc2, Scalar(1), tol)) << "Reference U(" << i << "): Column 2 is not normalized.";

			Scalar sqLenResUc0 = resU.col(0).squaredNorm();
			Scalar sqLenResUc1 = resU.col(1).squaredNorm();
			Scalar sqLenResUc2 = resU.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenResUc0, Scalar(1), tol)) << "Result U(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenResUc1, Scalar(1), tol)) << "Result U(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenResUc2, Scalar(1), tol)) << "Result U(" << i << "): Column 2 is not normalized.";

			Scalar sqLenRefVc0 = refV.col(0).squaredNorm();
			Scalar sqLenRefVc1 = refV.col(1).squaredNorm();
			Scalar sqLenRefVc2 = refV.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenRefVc0, Scalar(1), tol)) << "Reference V(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefVc1, Scalar(1), tol)) << "Reference V(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefVc2, Scalar(1), tol)) << "Reference V(" << i << "): Column 2 is not normalized.";

			Scalar sqLenResVc0 = resV.col(0).squaredNorm();
			Scalar sqLenResVc1 = resV.col(1).squaredNorm();
			Scalar sqLenResVc2 = resV.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenResVc0, Scalar(1), tol)) << "Result V(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenResVc1, Scalar(1), tol)) << "Result V(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenResVc2, Scalar(1), tol)) << "Result V(" << i << "): Column 2 is not normalized.";

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

			EXPECT_TRUE(eqS) << "S(" << i << ") -\nRef: " << refS.format(fmt) << ",\nRes: " << resS.format(fmt);
			EXPECT_TRUE(eqR) << "R(" << i << ") -\nRef: " << refR.format(fmt) << ",\nRes: " << resR.format(fmt);
		}
	}
}

#if defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)
template<typename WideScalar>
void runMcAdamsTest(float tol)
{
	using scalar_t  = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

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

		Vcl::Mathematics::McAdamsJacobiSVD(S, U, V, 5);
		
		resU.at<real_t>(i) = U;
		resV.at<real_t>(i) = V;
		resS.at<real_t>(i) = S.diagonal();
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refV, refS, resU, resV, resS);
}
#endif // defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)

template<typename WideScalar>
void runTwoSidedTest(float tol)
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

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
void runQRTest(float tol)
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

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

#ifdef VCL_VECTORIZE_SSE
TEST(SVD33, McAdamsSVDFloat)
{
	runMcAdamsTest<float>(1e-5f);
}

TEST(SVD33, McAdamsSVDFloat4)
{
	runMcAdamsTest<Vcl::float4>(1e-5f);
}
#endif // defined(VCL_VECTORIZE_SSE)

#ifdef VCL_VECTORIZE_AVX
TEST(SVD33, McAdamsSVDFloat8)
{
	runMcAdamsTest<Vcl::float8>(1e-5f);
}
#endif // defined VCL_VECTORIZE_AVX

TEST(SVD33, TwoSidedSVDFloat)
{
	runTwoSidedTest<float>(1e-4f);
}
TEST(SVD33, TwoSidedSVDFloat4)
{
	runTwoSidedTest<Vcl::float4>(1e-5f);
}
TEST(SVD33, TwoSidedSVDFloat8)
{
	runTwoSidedTest<Vcl::float8>(1e-5f);
}

TEST(SVD33, QRSVDFloat)
{
	runQRTest<float>(1e-5f);
}
TEST(SVD33, QRSVDFloat4)
{
	runQRTest<Vcl::float4>(1e-5f);
}
TEST(SVD33, QRSVDFloat8)
{
	runQRTest<Vcl::float8>(1e-5f);
}
