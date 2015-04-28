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

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/math/jacobisvd33_mcadams.h>
#include <vcl/math/jacobisvd33_qr.h>
#include <vcl/math/jacobisvd33_twosided.h>
#include <vcl/util/precisetimer.h>

void perfEigenSVD
(
	int nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
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
		auto U = resU.at<float>(i);
		auto V = resV.at<float>(i);
		auto S = resS.at<float>(i);
		
		// Compute using Eigen
		Eigen::Matrix3f A = F.at<float>(i);
		Eigen::JacobiSVD<Eigen::Matrix3f> eigen_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

		// Store results
		U = eigen_svd.matrixU();
		V = eigen_svd.matrixV();
		S = eigen_svd.singularValues();
	}
	timer.stop();
	std::cout << "Eigen Jacobi SVD: " << timer.interval() / nr_problems * 1e9 << "[ns]" << std::endl;	
}

template<typename WideScalar>
void perfTwoSidedSVD
(
	int nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
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
		auto V = resV.at<real_t>(i);
		auto S = resS.at<real_t>(i);
		
		// Compute SVD using 2-sided Jacobi iterations (Brent)
		matrix3_t SV = F.at<real_t>(i);
		matrix3_t matU = matrix3_t::Identity();
		matrix3_t matV = matrix3_t::Identity();

		avg_nr_iter += Vcl::Mathematics::TwoSidedJacobiSVD(SV, matU, matV, false);

		// Store results
		U = matU;
		V = matV;
		S = SV.diagonal();
	}
	timer.stop();
	std::cout << "Two-sided Jacobi SVD (Brent): " << timer.interval() / nr_problems * 1e9 << "[ns], Avg. iterations: " << (double) (avg_nr_iter * width) / (double) nr_problems << std::endl;
}
	
template<typename WideScalar>
void perfJacobiSVDQR
(
	int nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
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
		auto V = resV.at<real_t>(i);
		auto S = resS.at<real_t>(i);

		// Compute SVD using Jacobi iterations and QR decomposition
		matrix3_t SV = F.at<real_t>(i);
		matrix3_t matU = matrix3_t::Identity();
		matrix3_t matV = matrix3_t::Identity();

		avg_nr_iter += Vcl::Mathematics::QRJacobiSVD(SV, matU, matV);

		// Store results
		U = matU;
		V = matV;
		S = SV.diagonal();
	}
	timer.stop();
	std::cout << "Jacobi SVD (Symm. EV, QR): " << timer.interval() / nr_problems * 1e9 << "[ns], Avg. iterations: " << (double) (avg_nr_iter * width) / (double) nr_problems << std::endl;
}

template<typename WideScalar>
void perfMcAdamsSVD
(
	int nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
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
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(nr_problems / width); i++)
	{
		// Map data
		auto U = resU.at<real_t>(i);
		auto V = resV.at<real_t>(i);
		auto S = resS.at<real_t>(i);

		// Compute SVD using Jacobi iterations and QR decomposition
		matrix3_t SV = F.at<real_t>(i);
		matrix3_t matU = matrix3_t::Identity();
		matrix3_t matV = matrix3_t::Identity();

		avg_nr_iter += Vcl::Mathematics::McAdamsJacobiSVD(SV, matU, matV);

		// Store results
		U = matU;
		V = matV;
		S = SV.diagonal();
	}
	timer.stop();
	std::cout << "Jacobi SVD (McAdams): " << timer.interval() / nr_problems * 1e9 << "[ns], Avg. iterations: " << (double) (avg_nr_iter * width) / (double) nr_problems << std::endl;
}
	
int main(int, char**)
{
	size_t nr_problems = 1024*1024;

	Vcl::Core::InterleavedArray<float, 3, 3, -1> F(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resV(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(nr_problems);

	// Initialize data
	for (int i = 0; i < (int) nr_problems; i++)
	{
		F.at<float>(i).setRandom();
	}
	
	// Test Performance: Eigen Jacobi SVD
	perfEigenSVD(nr_problems, F, resU, resV, resS);
	
	// Test Performance: Two-sided Jacobi SVD (Brent)
	perfTwoSidedSVD<float>(nr_problems, F, resU, resV, resS);
	perfTwoSidedSVD<Vcl::float4>(nr_problems, F, resU, resV, resS);
	perfTwoSidedSVD<Vcl::float8>(nr_problems, F, resU, resV, resS);
	
	// Test Performance: Jacobi SVD with symmetric EV computation and QR decomposition
	perfJacobiSVDQR<float>(nr_problems, F, resU, resV, resS);
	perfJacobiSVDQR<Vcl::float4>(nr_problems, F, resU, resV, resS);
	perfJacobiSVDQR<Vcl::float8>(nr_problems, F, resU, resV, resS);
	
	// Test Performance: McAdams SVD solver
	perfMcAdamsSVD<float>(nr_problems, F, resU, resV, resS);
	perfMcAdamsSVD<Vcl::float4>(nr_problems, F, resU, resV, resS);
	perfMcAdamsSVD<Vcl::float8>(nr_problems, F, resU, resV, resS);
}
