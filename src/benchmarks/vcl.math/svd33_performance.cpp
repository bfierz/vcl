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

// Google benchmark
#include "benchmark/benchmark.h"

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

#ifdef VCL_CUDA_SUPPORT
template<typename WideScalar>
void perfCudaMcAdamsSVD
(
	size_t nr_problems,
	unsigned int iters,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS
)
{
	using namespace Vcl::Compute::Cuda;

	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	Platform::initialise();
	auto& dev = Platform::instance()->device(0);
	auto ctx = Vcl::Core::make_owner<Context>(dev);

	Vcl::Mathematics::Cuda::JacobiSVD33 solver(ctx);

	auto queue = Vcl::Core::dynamic_pointer_cast<CommandQueue>(ctx->defaultQueue());

	Vcl::Util::PreciseTimer timer;
	timer.start();

	// Solve the SVDs
	solver(*queue, F, resU, resV, resS);

	queue->sync();
	timer.stop();
	std::cout << "Jacobi SVD (McAdams) - CUDA - " << iters << " Iterations: " << timer.interval() / nr_problems * 1e9 << "[ns]" << std::endl;
}
#endif // defined VCL_CUDA_SUPPORT

#ifdef VCL_OPENCL_SUPPORT
template<typename WideScalar>
void perfOpenCLMcAdamsSVD
(
	size_t nr_problems,
	unsigned int iters,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resV,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS
)
{
	using namespace Vcl::Compute::OpenCL;

	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	Platform::initialise();
	auto& dev = Platform::instance()->device(0);
	auto ctx = Vcl::Core::make_owner<Context>(dev);

	Vcl::Mathematics::OpenCL::JacobiSVD33 solver(ctx);

	auto queue = Vcl::Core::dynamic_pointer_cast<CommandQueue>(ctx->defaultQueue());

	Vcl::Util::PreciseTimer timer;
	timer.start();

	// Solve the SVDs
	solver(*queue, F, resU, resV, resS);

	queue->sync();
	timer.stop();
	std::cout << "Jacobi SVD (McAdams) - OpenCL - " << iters << " Iterations: " << timer.interval() / nr_problems * 1e9 << "[ns]" << std::endl;
}
#endif // defined VCL_OPENCL_SUPPORT

/*int main(int, char**)
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
	perfTwoSidedSVD<Vcl::float16>(nr_problems, F, resU, resV, resS);

	// Test Performance: Jacobi SVD with symmetric EV computation and QR decomposition
	perfJacobiSVDQR<float>(nr_problems, F, resU, resV, resS);
	perfJacobiSVDQR<Vcl::float4>(nr_problems, F, resU, resV, resS);
	perfJacobiSVDQR<Vcl::float8>(nr_problems, F, resU, resV, resS);
	perfJacobiSVDQR<Vcl::float16>(nr_problems, F, resU, resV, resS);

	// Test Performance: McAdams SVD solver
	perfMcAdamsSVD<float>(nr_problems, 4, F, resU, resV, resS);
	perfMcAdamsSVD<Vcl::float4>(nr_problems, 4, F, resU, resV, resS);

#ifdef VCL_VECTORIZE_AVX
	perfMcAdamsSVD<Vcl::float8>(nr_problems, 4, F, resU, resV, resS);
#endif // defined VCL_VECTORIZE_AVX

	perfMcAdamsSVD<float>(nr_problems, 5, F, resU, resV, resS);
	perfMcAdamsSVD<Vcl::float4>(nr_problems, 5, F, resU, resV, resS);

#ifdef VCL_VECTORIZE_AVX
	perfMcAdamsSVD<Vcl::float8>(nr_problems, 5, F, resU, resV, resS);
#endif // defined VCL_VECTORIZE_AVX

#ifdef VCL_CUDA_SUPPORT
	perfCudaMcAdamsSVD<float>(nr_problems, 4, F, resU, resV, resS);
#endif // defined VCL_CUDA_SUPPORT

#ifdef VCL_OPENCL_SUPPORT
	perfOpenCLMcAdamsSVD<float>(nr_problems, 4, F, resU, resV, resS);
#endif // defined VCL_OPENCL_SUPPORT

	return 0;
}*/

// Global data store for one time problem setup
const size_t nr_problems = 8192;

Vcl::Core::InterleavedArray<float, 3, 3, -1> F(nr_problems);

void perfEigenSVD(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resV(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(state.range(0));

	for (auto _ : state)
	{
		for (int i = 0; i < state.range(0); ++i)
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
	}

	benchmark::DoNotOptimize(resU);
	benchmark::DoNotOptimize(resV);
	benchmark::DoNotOptimize(resS);

	state.SetItemsProcessed(state.iterations() * state.range(0));
}

template<typename WideScalar>
void perfTwoSidedSVD(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resV(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(state.range(0));

	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	for (auto _ : state)
	{
		for (size_t i = 0; i < state.range(0) / width; i++)
		{
			// Map data
			auto U = resU.at<real_t>(i);
			auto V = resV.at<real_t>(i);
			auto S = resS.at<real_t>(i);

			// Compute SVD using 2-sided Jacobi iterations (Brent)
			matrix3_t SV = F.at<real_t>(i);
			matrix3_t matU = matrix3_t::Identity();
			matrix3_t matV = matrix3_t::Identity();

			Vcl::Mathematics::TwoSidedJacobiSVD(SV, matU, matV, false);

			// Store results
			U = matU;
			V = matV;
			S = SV.diagonal();
		}
	}

	benchmark::DoNotOptimize(resU);
	benchmark::DoNotOptimize(resV);
	benchmark::DoNotOptimize(resS);

	state.SetItemsProcessed(state.iterations() * state.range(0));
}

template<typename WideScalar>
void perfJacobiSVDQR(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resV(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(state.range(0));

	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	for (auto _ : state)
	{
		for (size_t i = 0; i < state.range(0) / width; i++)
		{
			// Map data
			auto U = resU.at<real_t>(i);
			auto V = resV.at<real_t>(i);
			auto S = resS.at<real_t>(i);

			// Compute SVD using Jacobi iterations and QR decomposition
			matrix3_t SV = F.at<real_t>(i);
			matrix3_t matU = matrix3_t::Identity();
			matrix3_t matV = matrix3_t::Identity();

			Vcl::Mathematics::QRJacobiSVD(SV, matU, matV);

			// Store results
			U = matU;
			V = matV;
			S = SV.diagonal();
		}
	}

	benchmark::DoNotOptimize(resU);
	benchmark::DoNotOptimize(resV);
	benchmark::DoNotOptimize(resS);

	state.SetItemsProcessed(state.iterations() * state.range(0));
}

template<typename WideScalar, int Iters>
void perfMcAdamsSVD(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resV(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(state.range(0));

	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	for (auto _ : state)
	{
		for (size_t i = 0; i < state.range(0) / width; i++)
		{
			// Map data
			auto U = resU.at<real_t>(i);
			auto V = resV.at<real_t>(i);
			auto S = resS.at<real_t>(i);

			// Compute SVD using Jacobi iterations and QR decomposition
			matrix3_t SV = F.at<real_t>(i);
			matrix3_t matU = matrix3_t::Identity();
			matrix3_t matV = matrix3_t::Identity();

			Vcl::Mathematics::McAdamsJacobiSVD(SV, matU, matV, Iters);

			// Store results
			U = matU;
			V = matV;
			S = SV.diagonal();
		}
	}

	benchmark::DoNotOptimize(resU);
	benchmark::DoNotOptimize(resV);
	benchmark::DoNotOptimize(resS);

	state.SetItemsProcessed(state.iterations() * state.range(0));
}

using Vcl::float16;
using Vcl::float4;
using Vcl::float8;

// Test Performance: Eigen Jacobi SVD
BENCHMARK(perfEigenSVD)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);

// Test Performance: Two-sided Jacobi SVD (Brent)
BENCHMARK_TEMPLATE(perfTwoSidedSVD, float)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfTwoSidedSVD, float4)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfTwoSidedSVD, float8)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfTwoSidedSVD, float16)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);

// Test Performance: Jacobi SVD with symmetric EV computation and QR decomposition
BENCHMARK_TEMPLATE(perfJacobiSVDQR, float)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobiSVDQR, float4)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobiSVDQR, float8)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobiSVDQR, float16)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);

// Test Performance: McAdams SVD solver
BENCHMARK_TEMPLATE2(perfMcAdamsSVD, float, 4)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE2(perfMcAdamsSVD, float4, 4)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
#ifdef VCL_VECTORIZE_AVX
BENCHMARK_TEMPLATE2(perfMcAdamsSVD, float8, 4)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
#endif // defined VCL_VECTORIZE_AVX

BENCHMARK_TEMPLATE2(perfMcAdamsSVD, float, 5)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE2(perfMcAdamsSVD, float4, 5)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
#ifdef VCL_VECTORIZE_AVX
BENCHMARK_TEMPLATE2(perfMcAdamsSVD, float8, 5)->Arg(128)->Arg(512)->Arg(8192)->ThreadRange(1, 16);
#endif // defined VCL_VECTORIZE_AVX

int main(int argc, char** argv)
{
	// Initialize data
	createRandomProblems(nr_problems, F);

	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
