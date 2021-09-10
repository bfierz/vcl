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

// Google benchmark
#include "benchmark/benchmark.h"

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/math/jacobieigen33_selfadjoint.h>
#include <vcl/math/jacobieigen33_selfadjoint_quat.h>
#include <vcl/util/precisetimer.h>

#include "problems.h"

// Global data store for one time problem setup
const size_t nr_problems = 1024 * 1024;

// Problem set
Vcl::Core::InterleavedArray<float, 3, 3, -1> F(nr_problems);

void perfEigenIterative(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(state.range(0));

	for (auto _ : state)
	{
		for (int i = 0; i < state.range(0); ++i)
		{
			Vcl::Matrix3f A = F.at<float>(i);

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
			solver.compute(A, Eigen::ComputeEigenvectors);

			resU.at<float>(i) = solver.eigenvectors();
			resS.at<float>(i) = solver.eigenvalues();
		}
	}

	state.counters["Iterations"] = 0;
	benchmark::DoNotOptimize(resU);
	benchmark::DoNotOptimize(resS);

	state.SetItemsProcessed(state.iterations() * state.range(0));
}

void perfEigenDirect(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(state.range(0));

	for (auto _ : state)
	{
		for (int i = 0; i < state.range(0); ++i)
		{
			Vcl::Matrix3f A = F.at<float>(i);

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
			solver.computeDirect(A, Eigen::ComputeEigenvectors);

			resU.at<float>(i) = solver.eigenvectors();
			resS.at<float>(i) = solver.eigenvalues();
		}
	}

	state.counters["Iterations"] = 0;
	benchmark::DoNotOptimize(resU);
	benchmark::DoNotOptimize(resS);

	state.SetItemsProcessed(state.iterations() * state.range(0));
}

template<typename WideScalar>
void perfJacobi(benchmark::State& state)
{
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(state.range(0));

	int avg_nr_iter = 0;
	for (auto _ : state)
	{
		avg_nr_iter = 0;
		for (int i = 0; i < state.range(0) / width; ++i)
		{
			matrix3_t A = F.at<real_t>(i);
			matrix3_t U = matrix3_t::Identity();

			avg_nr_iter += Vcl::Mathematics::SelfAdjointJacobiEigen(A, U);

			resU.at<real_t>(i) = U;
			resS.at<real_t>(i) = A.diagonal();
		}
	}

	state.counters["Iterations"] = (double)(avg_nr_iter * width) / (double)state.range(0);
	benchmark::DoNotOptimize(resU);
	benchmark::DoNotOptimize(resS);

	state.SetItemsProcessed(state.iterations() * state.range(0));
}

template<typename WideScalar>
void perfJacobiQuat(benchmark::State& state)
{
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	Vcl::Core::InterleavedArray<float, 3, 3, -1> resU(state.range(0));
	Vcl::Core::InterleavedArray<float, 3, 1, -1> resS(state.range(0));

	int avg_nr_iter = 0;
	for (auto _ : state)
	{
		avg_nr_iter = 0;
		for (int i = 0; i < state.range(0) / width; ++i)
		{
			matrix3_t A = F.at<real_t>(i);
			matrix3_t U = matrix3_t::Identity();

			avg_nr_iter += Vcl::Mathematics::SelfAdjointJacobiEigenQuat(A, U);

			resU.at<real_t>(i) = U;
			resS.at<real_t>(i) = A.diagonal();
		}
	}

	state.counters["Iterations"] = (double)(avg_nr_iter * width) / (double)state.range(0);
	benchmark::DoNotOptimize(resU);
	benchmark::DoNotOptimize(resS);

	state.SetItemsProcessed(state.iterations() * state.range(0));
}

using Vcl::float16;
using Vcl::float4;
using Vcl::float8;

BENCHMARK(perfEigenIterative)->Arg(128); // ->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK(perfEigenDirect)->Arg(128);    // ->Arg(512)->Arg(8192)->ThreadRange(1, 16);

BENCHMARK_TEMPLATE(perfJacobi, float)->Arg(128);   //->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobi, float4)->Arg(128);  //->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobi, float8)->Arg(128);  //->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobi, float16)->Arg(128); //->Arg(512)->Arg(8192)->ThreadRange(1, 16);

BENCHMARK_TEMPLATE(perfJacobiQuat, float)->Arg(128);   //->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobiQuat, float4)->Arg(128);  //->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobiQuat, float8)->Arg(128);  //->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfJacobiQuat, float16)->Arg(128); //->Arg(512)->Arg(8192)->ThreadRange(1, 16);

int main(int argc, char** argv)
{
	// Initialize data
	createSymmetricProblems(nr_problems, F);

	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
