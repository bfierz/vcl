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
#include <fstream>
#include <iostream>
#include <random>

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/math/jacobisvd33_mcadams.h>
#include <vcl/math/jacobisvd33_qr.h>
#include <vcl/math/jacobisvd33_twosided.h>

#ifdef VCL_CUDA_SUPPORT
#	include <vcl/compute/cuda/commandqueue.h>
#	include <vcl/compute/cuda/context.h>
#	include <vcl/compute/cuda/device.h>
#	include <vcl/compute/cuda/platform.h>
#	include <vcl/math/cuda/jacobisvd33_mcadams.h>
#endif

#ifdef VCL_OPENCL_SUPPORT
#	include <vcl/compute/opencl/commandqueue.h>
#	include <vcl/compute/opencl/context.h>
#	include <vcl/compute/opencl/device.h>
#	include <vcl/compute/opencl/platform.h>
#	include <vcl/math/opencl/jacobisvd33_mcadams.h>
#endif

#include "problems.h"

template<typename Scalar>
void computeReferenceSolution(
	size_t nr_problems,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& U,
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& V,
	Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& S)
{
	// Compute reference using Eigen
	for (size_t i = 0; i < nr_problems; i++)
	{
		Vcl::Matrix3f A = F.template at<Scalar>(i);
		Eigen::JacobiSVD<Vcl::Matrix3f> eigen_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
		U.template at<Scalar>(i) = eigen_svd.matrixU();
		V.template at<Scalar>(i) = eigen_svd.matrixV();
		S.template at<Scalar>(i) = eigen_svd.singularValues();
	}
}

template<typename WideScalar, typename Func>
void computeSolution(
	size_t nr_problems,
	Func& func,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS)
{
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	int avg_nr_iter = 0;
	for (size_t i = 0; i < nr_problems / width; i++)
	{
		// Map data
		auto U = resU.at<real_t>(i);
		auto V = resV.at<real_t>(i);
		auto S = resS.at<real_t>(i);

		matrix3_t SV = F.at<real_t>(i);
		matrix3_t matU = matrix3_t::Identity();
		matrix3_t matV = matrix3_t::Identity();

		avg_nr_iter += func(SV, matU, matV);

		// Store results
		U = matU;
		V = matV;
		S = SV.diagonal();
	}
}

#ifdef VCL_CUDA_SUPPORT
void cudaMcAdamsSVD(
	size_t nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS)
{
	using namespace Vcl::Compute::Cuda;

	Platform::initialise();
	auto& dev = Platform::instance()->device(0);
	auto ctx = Vcl::Core::make_owner<Context>(dev);

	auto queue = Vcl::Core::dynamic_pointer_cast<CommandQueue>(ctx->defaultQueue());

	Vcl::Mathematics::Cuda::JacobiSVD33 solver(ctx);
	solver(*queue, F, resU, resV, resS);
	queue->sync();
}
#endif // defined VCL_CUDA_SUPPORT

#ifdef VCL_OPENCL_SUPPORT
void openCLMcAdamsSVD(
	size_t nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS)
{
	using namespace Vcl::Compute::OpenCL;

	Platform::initialise();
	auto& dev = Platform::instance()->device(0);
	auto ctx = Vcl::Core::make_owner<Context>(dev);

	auto queue = Vcl::Core::dynamic_pointer_cast<CommandQueue>(ctx->defaultQueue());

	Vcl::Mathematics::OpenCL::JacobiSVD33 solver(ctx);
	solver(*queue, F, resU, resV, resS);
	queue->sync();
}
#endif // defined VCL_OPENCL_SUPPORT

template<typename Scalar>
void checkSolution(
	const char* Name,
	const char* file,
	size_t nr_problems,
	Scalar tol,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refUa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refVa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& refSa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resUa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resVa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& resSa)
{
	using scalar_t = Scalar;

	int wrong_computations, wrong_v_computations, wrong_u_computations;
	scalar_t accum_error, accum_v_error, accum_u_error;
	std::ofstream fout;

	wrong_computations = 0;
	wrong_v_computations = 0;
	wrong_u_computations = 0;
	accum_error = 0;
	accum_v_error = 0;
	accum_u_error = 0;
	fout.open(file);

	for (int j = 0; j < (int)nr_problems; j++)
	{
		Vcl::Matrix3f refU = refUa.template at<scalar_t>(j);
		Vcl::Matrix3f refV = refVa.template at<scalar_t>(j);
		Vcl::Vector3f refS = refSa.template at<scalar_t>(j);
		Vcl::Matrix3f cU = resUa.template at<scalar_t>(j);
		Vcl::Matrix3f cV = resVa.template at<scalar_t>(j);
		Vcl::Vector3f cS = resSa.template at<scalar_t>(j);
		bool eqU = refU.array().abs().isApprox(cU.array().abs(), tol);
		bool eqV = refV.array().abs().isApprox(cV.array().abs(), tol);
		bool eqS = refS.array().abs().isApprox(cS.array().abs(), tol);

		if (!eqS || !eqU || !eqV)
			fout << j;

		if (!eqS)
		{
			wrong_computations++;
			scalar_t err = abs(refS.array().abs() - cS.array().abs()).maxCoeff();
			accum_error += err;
			fout << ", E: " << err;
		}
		if (!eqU)
		{
			wrong_u_computations++;
			scalar_t err = abs(refU.array().abs() - cU.array().abs()).maxCoeff();
			accum_u_error += err;
			fout << ", U: " << err;
		}
		if (!eqV)
		{
			wrong_v_computations++;
			scalar_t err = abs(refV.array().abs() - cV.array().abs()).maxCoeff();
			accum_v_error += err;
			fout << ", V: " << err;
		}
		if (!eqS || !eqU || !eqV)
			fout << std::endl;
	}

	fout.close();
	std::cout << Name << " - Errors: (" << wrong_computations << ", " << wrong_u_computations << ", " << wrong_v_computations << "), "
			  << "Avg. Singular value error: " << accum_error / std::max(wrong_computations, 1) << ", "
			  << "Avg. U error: " << accum_u_error / std::max(wrong_u_computations, 1) << ", "
			  << "Avg. V error: " << accum_v_error / std::max(wrong_v_computations, 1) << std::endl;
}

int main(int, char**)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::Mathematics::McAdamsJacobiSVD;
	using Vcl::Mathematics::QRJacobiSVD;
	using Vcl::Mathematics::TwoSidedJacobiSVD;

	const size_t nr_problems = 1024 * 1024;

	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resV(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<float, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 3, -1> refV(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> refS(nr_problems);

	Vcl::Core::InterleavedArray<float, 3, 3, -1> F(nr_problems);
	createRandomProblems(nr_problems, F);
	computeReferenceSolution(nr_problems, F, refU, refV, refS);
	
	using ComputeSvdFloat   = int (*)(Eigen::Matrix<float,   3, 3>&, Eigen::Matrix<float,   3, 3>&, Eigen::Matrix<float,   3, 3>&);
	using ComputeSvdFloat4  = int (*)(Eigen::Matrix<float4,  3, 3>&, Eigen::Matrix<float4,  3, 3>&, Eigen::Matrix<float4,  3, 3>&);
	using ComputeSvdFloat8  = int (*)(Eigen::Matrix<float8,  3, 3>&, Eigen::Matrix<float8,  3, 3>&, Eigen::Matrix<float8,  3, 3>&);
	using ComputeSvdFloat16 = int (*)(Eigen::Matrix<float16, 3, 3>&, Eigen::Matrix<float16, 3, 3>&, Eigen::Matrix<float16, 3, 3>&);

	// Test correctness: Two-sided Jacobi SVD (Brent)
	auto two_sided_float   = static_cast<ComputeSvdFloat>  (TwoSidedJacobiSVD);
	auto two_sided_float4  = static_cast<ComputeSvdFloat4> (TwoSidedJacobiSVD);
	auto two_sided_float8  = static_cast<ComputeSvdFloat8> (TwoSidedJacobiSVD);
	auto two_sided_float16 = static_cast<ComputeSvdFloat16>(TwoSidedJacobiSVD);
	computeSolution<float>  (nr_problems, two_sided_float,   F, resU, resV, resS); checkSolution("TwoSidedJacobiSVD - float",   "two_sided_jacobi_svd_float_errors.txt",   nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	computeSolution<float4> (nr_problems, two_sided_float4,  F, resU, resV, resS); checkSolution("TwoSidedJacobiSVD - float4",  "two_sided_jacobi_svd_float4_errors.txt",  nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	computeSolution<float8> (nr_problems, two_sided_float8,  F, resU, resV, resS); checkSolution("TwoSidedJacobiSVD - float8",  "two_sided_jacobi_svd_float8_errors.txt",  nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	computeSolution<float16>(nr_problems, two_sided_float16, F, resU, resV, resS); checkSolution("TwoSidedJacobiSVD - float16", "two_sided_jacobi_svd_float16_errors.txt", nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	
	// Test correctness: Jacobi SVD with symmetric EV computation and QR decomposition
	auto jacobi_float   = static_cast<ComputeSvdFloat>  (QRJacobiSVD);
	auto jacobi_float4  = static_cast<ComputeSvdFloat4> (QRJacobiSVD);
	auto jacobi_float8  = static_cast<ComputeSvdFloat8> (QRJacobiSVD);
	auto jacobi_float16 = static_cast<ComputeSvdFloat16>(QRJacobiSVD);
	computeSolution<float>  (nr_problems, jacobi_float,   F, resU, resV, resS); checkSolution("JacobiSVDQR - float",   "jacobi_svd_qr_float_errors.txt",   nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	computeSolution<float4> (nr_problems, jacobi_float4,  F, resU, resV, resS); checkSolution("JacobiSVDQR - float4",  "jacobi_svd_qr_float4_errors.txt",  nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	computeSolution<float8> (nr_problems, jacobi_float8,  F, resU, resV, resS); checkSolution("JacobiSVDQR - float8",  "jacobi_svd_qr_float8_errors.txt",  nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	computeSolution<float16>(nr_problems, jacobi_float16, F, resU, resV, resS); checkSolution("JacobiSVDQR - float16", "jacobi_svd_qr_float16_errors.txt", nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	
	// Test correctness: McAdams SVD solver
	auto mcadams_float   = static_cast<ComputeSvdFloat> (McAdamsJacobiSVD);
	auto mcadams_float4  = static_cast<ComputeSvdFloat4>(McAdamsJacobiSVD);
	computeSolution<float>  (nr_problems, mcadams_float,   F, resU, resV, resS); checkSolution("McAdamsSVD - float",   "mc_adams_svd_float_errors.txt",  nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	computeSolution<float4> (nr_problems, mcadams_float4,  F, resU, resV, resS); checkSolution("McAdamsSVD - float4",  "mc_adams_svd_float4_errors.txt", nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
	
#ifdef VCL_VECTORIZE_AVX
	auto mcadams_float8  = static_cast<ComputeSvdFloat8>(McAdamsJacobiSVD);
	computeSolution<float8> (nr_problems, mcadams_float8,  F, resU, resV, resS); checkSolution("McAdamnsSVD - float8", "mc_adams_svd_float8_errors.txt", nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
#endif // defined VCL_VECTORIZE_AVX

#ifdef VCL_CUDA_SUPPORT
	cudaMcAdamsSVD(nr_problems, F, resU, resV, resS); checkSolution("McAdamsSVD - CUDA", "cuda_mc_adams_svd_errors.txt", nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
#endif // defined VCL_CUDA_SUPPORT

#ifdef VCL_OPENCL_SUPPORT
	openCLMcAdamsSVD(nr_problems, F, resU, resV, resS); checkSolution("McAdamsSVD - OpenCL", "opencl_mc_adams_svd_errors.txt", nr_problems, 1e-5f, refU, refV, refS, resU, resV, resS);
#endif // defined VCL_OPENCL_SUPPORT
}
