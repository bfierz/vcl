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

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/math/polardecomposition.h>
#include <vcl/math/rotation33_torque.h>
#include <vcl/util/precisetimer.h>

// Google benchmark
#include "benchmark/benchmark.h"

// Global data store for one time problem setup
const size_t nr_problems = 8192;

Vcl::Core::InterleavedArray<float, 3, 3, -1> F_30(nr_problems);
Vcl::Core::InterleavedArray<float, 3, 3, -1> F_60(nr_problems);
Vcl::Core::InterleavedArray<float, 3, 3, -1> F(nr_problems);

template<typename Scalar>
void createProblems
(
	size_t nr_problems,
	Scalar max_angle,
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>* F
)
{
	// Random number generator
	std::mt19937_64 rng;
	std::uniform_real_distribution<float> d;
	std::uniform_real_distribution<float> a{ -max_angle, max_angle };

	// Initialize data
	for (int i = 0; i < (int)nr_problems; i++)
	{
		// Rest-state
		Eigen::Matrix<Scalar, 3, 3> X0;
		X0 << d(rng), d(rng), d(rng),
		      d(rng), d(rng), d(rng),
		      d(rng), d(rng), d(rng);

		// Rotation angle
		Scalar angle = a(rng);

		// Rotation axis
		Eigen::Matrix<Scalar, 3, 1> rot_vec;
		rot_vec << d(rng), d(rng), d(rng);
		rot_vec.normalize();

		// Rotation matrix
		Eigen::Matrix<Scalar, 3, 3> Rot = Eigen::AngleAxis<Scalar>{ angle, rot_vec }.toRotationMatrix();

		Eigen::Matrix<Scalar, 3, 3> X = Rot * X0;
		F->template at<Scalar>(i) = X * X0.inverse();
	}
}

void perfEigenSVD(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resR(state.range_x());

	while (state.KeepRunning())
	{
		for (int i = 0; i < state.range_x(); ++i)
		{
			// Map data
			auto R = resR.at<float>(i);

			// Compute using Eigen
			Eigen::Matrix3f A = F.at<float>(i);
			Eigen::JacobiSVD<Eigen::Matrix3f> eigen_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

			// Store results
			R = eigen_svd.matrixU() * eigen_svd.matrixV().transpose();
		}
	}

	benchmark::DoNotOptimize(resR);

	state.SetItemsProcessed(state.iterations() * state.range_x());
}

template<typename WideScalar>
void perfPolarDecomposition(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resR(state.range_x());

	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	while (state.KeepRunning())
	{
		for (size_t i = 0; i < state.range_x() / width; i++)
		{
			// Map data
			auto R = resR.at<real_t>(i);

			// Compute SVD using 2-sided Jacobi iterations (Brent)
			matrix3_t A = F.at<real_t>(i);
			matrix3_t RR= A;

			Vcl::Mathematics::PolarDecomposition(A, RR, nullptr);

			// Store results
			R = RR;
		}
	}

	benchmark::DoNotOptimize(resR);

	state.SetItemsProcessed(state.iterations() * state.range_x());
}

template<typename WideScalar>
void perfRotationTorque(benchmark::State& state)
{
	Vcl::Core::InterleavedArray<float, 3, 3, -1> resR(state.range_x());

	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	while (state.KeepRunning())
	{
		for (size_t i = 0; i < state.range_x() / width; i++)
		{
			// Map data
			auto R = resR.at<real_t>(i);

			// Compute Rotation using the torque based method
			matrix3_t A = F.at<real_t>(i);
			matrix3_t RR = A;

			Vcl::Mathematics::Rotation(A, RR);

			// Store results
			R = RR;
		}
	}

	benchmark::DoNotOptimize(resR);

	state.SetItemsProcessed(state.iterations() * state.range_x());
}

using Vcl::float4;
using Vcl::float8;
using Vcl::float16;

// Test Performance: Eigen Jacobi SVD
BENCHMARK(perfEigenSVD)->Arg(128);// ->Arg(512)->Arg(8192)->ThreadRange(1, 16);

// Test Performance: Iterative jacobi polar decomposion
BENCHMARK_TEMPLATE(perfPolarDecomposition, float)  ->Arg(128);//->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfPolarDecomposition, float4) ->Arg(128);//->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfPolarDecomposition, float8) ->Arg(128);//->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfPolarDecomposition, float16)->Arg(128);//->Arg(512)->Arg(8192)->ThreadRange(1, 16);

// Test Performance: Iterative rotation estimation
BENCHMARK_TEMPLATE(perfRotationTorque, float)  ->Arg(128);//->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfRotationTorque, float4) ->Arg(128);//->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfRotationTorque, float8) ->Arg(128);//->Arg(512)->Arg(8192)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE(perfRotationTorque, float16)->Arg(128);//->Arg(512)->Arg(8192)->ThreadRange(1, 16);


int main(int argc, char** argv)
{
	// Initialize data
	createProblems<float>(nr_problems, 90, &F);
	
	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
