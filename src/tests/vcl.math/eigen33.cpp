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

// Eigen library
#include <Eigen/Dense>

// Include the relevant parts from the library
#include <vcl/core/interleavedarray.h>
#include <vcl/math/math.h>
#include <vcl/math/jacobieigen33_selfadjoint.h>
#include <vcl/math/jacobieigen33_selfadjoint_quat.h>

#define VCL_MATH_SELFADJOINTJACOBI_USE_RSQRT
#define VCL_MATH_SELFADJOINTJACOBI_USE_RCP
#include <vcl/math/jacobieigen33_selfadjoint_impl.h>

// Google test
#include <gtest/gtest.h>

// Common functions
namespace
{
	template<typename REAL>
	void SortEigenvalues(Eigen::Matrix<REAL, 3, 1>& A, Eigen::Matrix<REAL, 3, 3>& B)
	{
		// Bubble sort
		bool swapped = true;
		int j = 0;

		while (swapped)
		{
			swapped = false;
			j++;

			for (int i = 0; i < 3 - j; i++)
			{
				if (A(i) > A(i + 1))
				{
					std::swap(A(i), A(i + 1));
					B.col(i).swap(B.col(i + 1));

					swapped = true;
				}
			}
		}
	}

	template<typename Scalar>
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> createProblems(size_t nr_problems)
	{
		// Random number generator
		std::mt19937_64 rng;
		std::uniform_real_distribution<float> d;

		Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> A(nr_problems);
	
		// Initialize data
		for (int i = 0; i < (int) nr_problems; i++)
		{
			Eigen::Matrix<Scalar, 3, 3> rnd;
			rnd << d(rng), d(rng), d(rng),
				   d(rng), d(rng), d(rng),
				   d(rng), d(rng), d(rng);
			A.at<Scalar>(i) = rnd.transpose() * rnd;
		}

		return std::move(A);
	}

	template<typename Scalar>
	void computeReferenceSolution
	(
		size_t nr_problems,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& ATA,
		Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& U,
		Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& S
	)
	{
		// Compute reference using Eigen
		for (int i = 0; i < static_cast<int>(nr_problems); i++)
		{
			Vcl::Matrix3f A = ATA.at<Scalar>(i);

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
			solver.compute(A, Eigen::ComputeEigenvectors);

			U.at<Scalar>(i) = solver.eigenvectors();
			S.at<Scalar>(i) = solver.eigenvalues();
		}
	}

	template<typename Scalar>
	void checkSolution
	(
		size_t nr_problems,
		Scalar tol,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refUa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& refSa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resUa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& resSa
	)
	{
		using Vcl::Mathematics::equal;

		Eigen::IOFormat fmt(6, 0, ", ", ";", "[", "]");

		for (int i = 0; i < static_cast<int>(nr_problems); i++)
		{
			Vcl::Matrix3f refU = refUa.at<Scalar>(i);
			Vcl::Vector3f refS = refSa.at<Scalar>(i);

			Vcl::Matrix3f resU = resUa.at<Scalar>(i);
			Vcl::Vector3f resS = resSa.at<Scalar>(i);
			SortEigenvalues(resS, resU);

			Scalar sqLenRefUc0 = refU.col(0).squaredNorm();
			Scalar sqLenRefUc1 = refU.col(1).squaredNorm();
			Scalar sqLenRefUc2 = refU.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenRefUc0, Scalar(1), tol)) << "Index: " << i << ", Reference U: Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefUc1, Scalar(1), tol)) << "Index: " << i << ", Reference U: Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefUc2, Scalar(1), tol)) << "Index: " << i << ", Reference U: Column 2 is not normalized.";

			Scalar sqLenResUc0 = resU.col(0).squaredNorm();
			Scalar sqLenResUc1 = resU.col(1).squaredNorm();
			Scalar sqLenResUc2 = resU.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenResUc0, Scalar(1), tol)) << "Index: " << i << ", Result U: Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenResUc1, Scalar(1), tol)) << "Index: " << i << ", Result U: Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenResUc2, Scalar(1), tol)) << "Index: " << i << ", Result U: Column 2 is not normalized.";

			bool eqS = refS.array().abs().isApprox(resS.array().abs(), tol);
			bool eqU = refU.array().abs().isApprox(resU.array().abs(), 2*tol);

			EXPECT_TRUE(eqS) << "Index: " << i << ", S(" << i << ") -\nRef: " << refS.format(fmt) << ",\nRes: " << resS.format(fmt);
			EXPECT_TRUE(eqU) << "Index: " << i << ", U(" << i << ") -\nRef: " << refU.format(fmt) << ",\nRes: " << resU.format(fmt);

			Vcl::Matrix3f I = resU.transpose() * resU;
			Vcl::Matrix3f Iref = Vcl::Matrix3f::Identity();

			EXPECT_TRUE(equal(Iref, I, tol)) << "Index: " << i << ", Result U^t U: not Identity.";
		}
	}
}

template<typename WideScalar>
void runJacobiEigen33Test(float tol)
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	auto A = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, A, refU, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t ATA = A.at<real_t>(i);
		matrix3_t U;
		vector3_t S;

		Vcl::Mathematics::SelfAdjointJacobiEigen(ATA, U);
		S = ATA.diagonal();

		resU.at<real_t>(i) = U;
		resS.at<real_t>(i) = S;
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refS, resU, resS);
}

template<typename WideScalar>
void runJacobiEigenQuat33Test(float tol)
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	auto A = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, A, refU, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t ATA = A.at<real_t>(i);
		matrix3_t U;
		vector3_t S;

		Vcl::Mathematics::SelfAdjointJacobiEigenQuat(ATA, U);
		S = ATA.diagonal();

		resU.at<real_t>(i) = U;
		resS.at<real_t>(i) = S;
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refS, resU, resS);
}

TEST(Eigen33, JacobiRotationAngleFloat4)
{
	Vcl::float4 a11{ 0.0636276379f, 0.620737076f, 0.930427194f, 1.56127894f };
	Vcl::float4 a12{ 0.221034527f,  0.599013686f, 0.525564313f, 1.41785979f };
	Vcl::float4 a22{ 1.57842779f,   0.872990131f, 0.716681361f, 1.48913455f };

	Vcl::float4 c{ -0.989938557f, -0.776543200f, 0.774360895f, 0.716042221f };
	Vcl::float4 s{  0.141497672f,  0.630063951f, 0.632744193f, 0.698056936f };

	auto cs = Vcl::Mathematics::JacobiRotationAngle(a11, a12, a22);

	EXPECT_TRUE(Vcl::all(Vcl::equal(c, cs(0), Vcl::float4(1e-5f)))) << "Cosinus was computed wrong: " << cs(0);
	EXPECT_TRUE(Vcl::all(Vcl::equal(s, cs(1), Vcl::float4(1e-5f)))) <<   "Sinus was computed wrong: " << cs(1);
}

TEST(Eigen33, JacobiRotationAngleFloat8)
{
	Vcl::float8 a11{ 0.0636276379f, 0.620737076f, 0.930427194f, 1.56127894f, 1.36132240f,  1.17526937f,  1.15197527f,  1.03158689f  };
	Vcl::float8 a12{ 0.221034527f,  0.599013686f, 0.525564313f, 1.41785979f, 0.277331233f, 0.342120767f, 0.592817068f, 0.618788481f };
	Vcl::float8 a22{ 1.57842779f,   0.872990131f, 0.716681361f, 1.48913455f, 0.114264831f, 0.201089293f, 0.925793409f, 0.934002161f };

	Vcl::float8 c{ -0.989938557f, -0.776543200f, 0.774360895f, 0.716042221f, 0.978186727f, 0.953498065f, 0.770515859f, 0.734373033f };
	Vcl::float8 s{  0.141497672f,  0.630063951f, 0.632744193f, 0.698056936f, 0.207727432f, 0.301399142f, 0.637420893f, 0.678746164f };

	auto cs = Vcl::Mathematics::JacobiRotationAngle(a11, a12, a22);

	EXPECT_TRUE(Vcl::all(Vcl::equal(c, cs(0), Vcl::float8(1e-5f)))) << "Cosinus was computed wrong: " << cs(0);
	EXPECT_TRUE(Vcl::all(Vcl::equal(s, cs(1), Vcl::float8(1e-5f)))) <<   "Sinus was computed wrong: " << cs(1);
}

TEST(Eigen33, EigenFloat)
{
	runJacobiEigen33Test<float>(1e-5f);
}
TEST(Eigen33, EigenFloat4)
{
	runJacobiEigen33Test<Vcl::float4>(1e-5f);
}
TEST(Eigen33, EigenFloat8)
{
	runJacobiEigen33Test<Vcl::float8>(1e-5f);
}
TEST(Eigen33, EigenFloat16)
{
	runJacobiEigen33Test<Vcl::float16>(1e-5f);
}

TEST(Eigen33, EigenQuatFloat)
{
	runJacobiEigenQuat33Test<float>(1e-5f);
}
TEST(Eigen33, EigenQuatFloat4)
{
	runJacobiEigenQuat33Test<Vcl::float4>(1e-5f);
}
TEST(Eigen33, EigenQuatFloat8)
{
	runJacobiEigenQuat33Test<Vcl::float8>(1e-5f);
}
TEST(Eigen33, EigenQuatFloat16)
{
	runJacobiEigenQuat33Test<Vcl::float16>(1e-5f);
}
