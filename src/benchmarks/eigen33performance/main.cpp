/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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
#include <random>

// Eigen library
#include <Eigen/Dense>

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/math/jacobieigen33_selfadjoint.h>
#include <vcl/math/jacobieigen33_selfadjoint_quat.h>
#include <vcl/util/precisetimer.h>

template<typename Scalar>
Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> createProblems(size_t nr_problems)
{
	// Random number generator
	std::mt19937_64 rng;
	std::uniform_real_distribution<float> d;

	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> F(nr_problems);

	// Initialize data
	for (int i = 0; i < (int) nr_problems; i++)
	{
		Eigen::Matrix<Scalar, 3, 3> rnd;
		rnd << d(rng), d(rng), d(rng),
			d(rng), d(rng), d(rng),
			d(rng), d(rng), d(rng);
		F.at<Scalar>(i) = rnd.transpose() * rnd;
	}

	return std::move(F);
}

void perfEigenEigen
(
	int nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS
)
{
	Vcl::Util::PreciseTimer timer;
	timer.start();
#ifdef _OPENMP
#	pragma omp parallel for
#endif /* _OPENMP */
	for (int i = 0; i < (int) nr_problems; i++)
	{
		// Map data
		Vcl::Matrix3f A = F.at<float>(i);

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
		solver.compute(A, Eigen::ComputeEigenvectors);

		resU.at<float>(i) = solver.eigenvectors();
		resS.at<float>(i) = solver.eigenvalues();
	}
	timer.stop();
	std::cout << "Eigen Jacobi SVD: " << timer.interval() / nr_problems * 1e9 << "[ns]" << std::endl;	
}

void perfEigenEigenDirect
(
	int nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS
)
{
	Vcl::Util::PreciseTimer timer;
	timer.start();
#ifdef _OPENMP
#	pragma omp parallel for
#endif /* _OPENMP */
	for (int i = 0; i < (int) nr_problems; i++)
	{
		// Map data
		Vcl::Matrix3f A = F.at<float>(i);

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
		solver.computeDirect(A, Eigen::ComputeEigenvectors);

		resU.at<float>(i) = solver.eigenvectors();
		resS.at<float>(i) = solver.eigenvalues();
	}
	timer.stop();
	std::cout << "Eigen Jacobi SVD: " << timer.interval() / nr_problems * 1e9 << "[ns]" << std::endl;	
}

template<typename WideScalar>
void perfJacobiEigen
(
	int nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS
)
{
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t width = sizeof(real_t) / sizeof(float);
	
	Vcl::Util::PreciseTimer timer;
	timer.start();
	int avg_nr_iter = 0;
#ifdef _OPENMP
#	pragma omp parallel for
#endif /* _OPENMP */
	for (int i = 0; i < static_cast<int>(nr_problems / width); i++)
	{
		// Map data
		auto U = resU.at<real_t>(i);
		auto S = resS.at<real_t>(i);
		
		// Compute SVD using 2-sided Jacobi iterations (Brent)
		matrix3_t SV = F.at<real_t>(i);
		matrix3_t matU = matrix3_t::Identity();

		avg_nr_iter += Vcl::Mathematics::SelfAdjointJacobiEigen(SV, matU);

		// Store results
		U = matU;
		S = SV.diagonal();
	}
	timer.stop();
	std::cout << "Self-adjoint Jacobi Eigen Decomposition: " << timer.interval() / nr_problems * 1e9 << "[ns], Avg. iterations: " << (double) (avg_nr_iter * width) / (double) nr_problems << std::endl;
}
	
template<typename WideScalar>
void perfJacobiEigenQuat
(
	int nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS
)
{
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t width = sizeof(real_t) / sizeof(float);
	
	Vcl::Util::PreciseTimer timer;
	timer.start();
	int avg_nr_iter = 0;
#ifdef _OPENMP
#	pragma omp parallel for
#endif /* _OPENMP */
	for (int i = 0; i < static_cast<int>(nr_problems / width); i++)
	{
		// Map data
		auto U = resU.at<real_t>(i);
		auto S = resS.at<real_t>(i);

		// Compute SVD using Jacobi iterations and QR decomposition
		matrix3_t SV = F.at<real_t>(i);
		matrix3_t matU = matrix3_t::Identity();

		avg_nr_iter += Vcl::Mathematics::SelfAdjointJacobiEigenQuat(SV, matU);

		// Store results
		U = matU;
		S = SV.diagonal();
	}
	timer.stop();
	std::cout << "Self-adjoint Jacobi Quaternion Eigen Decomposition: " << timer.interval() / nr_problems * 1e9 << "[ns], Avg. iterations: " << (double) (avg_nr_iter * width) / (double) nr_problems << std::endl;
}
int main(int, char**)
{
	size_t nr_problems = 1024*1024;

	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(nr_problems);

	// Initialize data
	auto F = createProblems<float>(nr_problems);
	
	// Test Performance: Eigen Jacobi Decomposition
	perfEigenEigen(nr_problems, F, resU, resS);
	perfEigenEigenDirect(nr_problems, F, resU, resS);
	
	// Test Performance: Jacobi Eigenvalue Decomposition
	perfJacobiEigen<float>(nr_problems, F, resU, resS);
	perfJacobiEigen<Vcl::float4>(nr_problems, F, resU, resS);
	perfJacobiEigen<Vcl::float8>(nr_problems, F, resU, resS);
	perfJacobiEigen<Vcl::float16>(nr_problems, F, resU, resS);
	
	// Test Performance: Jacobi Eigenvalue Decomposition using quaternions
	perfJacobiEigenQuat<float>(nr_problems, F, resU, resS);
	perfJacobiEigenQuat<Vcl::float4>(nr_problems, F, resU, resS);
	perfJacobiEigenQuat<Vcl::float8>(nr_problems, F, resU, resS);
	perfJacobiEigenQuat<Vcl::float16>(nr_problems, F, resU, resS);
}
