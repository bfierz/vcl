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

// Google benchmark
#include "benchmark/benchmark.h"

// Include the relevant parts from the library
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/geometry/distancePoint3Triangle3.h>
#include <vcl/geometry/distanceTriangle3Triangle3.h>
#include <vcl/math/math.h>

// Reference code
#include <Mathematics/GteDistPointTriangle.h>
#include <Mathematics/GteDistTriangle3Triangle3.h>

// Tests the distance functions.
gte::Vector3<float> cast(const Eigen::Vector3f& vec)
{
	return{ vec.x(), vec.y(), vec.z() };
}

Eigen::Vector3f cast(const gte::Vector3<float>& vec)
{
	return{ vec[0], vec[1], vec[2] };
}

////////////////////////////////////////////////////////////////////////////////
// Triangle-Triangle distance
////////////////////////////////////////////////////////////////////////////////

void BM_TriTriEberly(benchmark::State& state)
{
	const int problem_size = 64;

	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_a(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_b(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_c(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_A(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_B(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_C(problem_size);

	for (int i = 0; i < problem_size; i++)
	{
		points_a.at<float>(i) = Eigen::Vector3f::Random();
		points_b.at<float>(i) = Eigen::Vector3f::Random();
		points_c.at<float>(i) = Eigen::Vector3f::Random();
		points_A.at<float>(i) = Eigen::Vector3f::Random();
		points_B.at<float>(i) = Eigen::Vector3f::Random();
		points_C.at<float>(i) = Eigen::Vector3f::Random();
	}

	// Compute the reference solution
	gte::DCPQuery<float, gte::Triangle3<float>, gte::Triangle3<float>> gteQuery;

	while (state.KeepRunning())
	{
		for (int i = 0; i < problem_size; i++)
		{
			Eigen::Vector3f triA_0 = points_a.at<float>(i);
			Eigen::Vector3f triA_1 = points_b.at<float>(i);
			Eigen::Vector3f triA_2 = points_c.at<float>(i);
			Eigen::Vector3f triB_0 = points_A.at<float>(i);
			Eigen::Vector3f triB_1 = points_B.at<float>(i);
			Eigen::Vector3f triB_2 = points_C.at<float>(i);

			gte::Triangle3<float> A{ cast(triA_0), cast(triA_1), cast(triA_2) };
			gte::Triangle3<float> B{ cast(triB_0), cast(triB_1), cast(triB_2) };

			benchmark::DoNotOptimize(gteQuery(A, B));
		}
	}

	state.SetItemsProcessed(problem_size * state.iterations());
}

template<typename Real, typename Int>
void BM_TriTri(benchmark::State& state)
{
	using namespace Vcl::Geometry;
	using Vcl::Mathematics::equal;
	
	using real_t = Real;
	using int_t = Int;

	using vector3_t = Eigen::Matrix<real_t, 3, 1>;
	using vector3i_t = Eigen::Matrix<int_t, 3, 1>;

	const int width = sizeof(real_t) / sizeof(float);
	const int problem_size = 64;

	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_a(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_b(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_c(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_A(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_B(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_C(problem_size);

	for (int i = 0; i < problem_size; i++)
	{
		points_a.at<float>(i) = Eigen::Vector3f::Random();
		points_b.at<float>(i) = Eigen::Vector3f::Random();
		points_c.at<float>(i) = Eigen::Vector3f::Random();
		points_A.at<float>(i) = Eigen::Vector3f::Random();
		points_B.at<float>(i) = Eigen::Vector3f::Random();
		points_C.at<float>(i) = Eigen::Vector3f::Random();
	}

	while (state.KeepRunning())
	{
		for (int i = 0; i < problem_size / width; i++)
		{
			vector3_t triA_0 = points_a.at<real_t>(i);
			vector3_t triA_1 = points_b.at<real_t>(i);
			vector3_t triA_2 = points_c.at<real_t>(i);
			vector3_t triB_0 = points_A.at<real_t>(i);
			vector3_t triB_1 = points_B.at<real_t>(i);
			vector3_t triB_2 = points_C.at<real_t>(i);

			vector3_t a, b;
			benchmark::DoNotOptimize(distance({ triA_0, triA_1, triA_2 }, { triB_0, triB_1, triB_2 }, a, b));
		}
	}

	state.SetItemsProcessed(problem_size * state.iterations());
}

// Register the function as a benchmark
BENCHMARK(BM_TriTriEberly)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE2(BM_TriTri, Vcl::float4, Vcl::int4)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE2(BM_TriTri, Vcl::float8, Vcl::int8)->ThreadRange(1, 16);
BENCHMARK_TEMPLATE2(BM_TriTri, Vcl::float16, Vcl::int16)->ThreadRange(1, 16);

////////////////////////////////////////////////////////////////////////////////


BENCHMARK_MAIN()
