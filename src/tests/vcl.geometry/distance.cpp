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
#include <vcl/config/eigen.h>

// C++ Standard Library
#include <random>
#include <vector>

// Include the relevant parts from the library
#include <vcl/core/memory/allocator.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/geometry/distance_ray3ray3.h>
#include <vcl/geometry/distancePoint3Triangle3.h>
#include <vcl/geometry/distanceTriangle3Triangle3.h>
#include <vcl/math/math.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Reference code
#include "ref/GteDistPointTriangle.h"
#include "ref/GteDistRayRay.h"
#include "ref/GteDistTriangle3Triangle3.h"

#include "distance_ref.h"

// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

// Tests the distance functions.
gte::Vector3<float> cast(const Eigen::Vector3f& vec)
{
	return{ vec.x(), vec.y(), vec.z() };
}

Eigen::Vector3f cast(const gte::Vector3<float>& vec)
{
	return{ vec[0], vec[1], vec[2] };
}

TEST(PointTriangleDistance, Simple)
{
	using Vcl::Geometry::distance;
	using Vcl::Mathematics::equal;

	//typedef Vcl::float16 real_t;
	typedef Vcl::float8 real_t;
	//typedef Vcl::float4 real_t;
	//typedef float real_t;

	typedef Eigen::Matrix<real_t, 3, 1> vector3_t;

	using WideVector = std::vector<real_t, Vcl::Core::Allocator<real_t, Vcl::Core::AlignedAllocPolicy<real_t, 32>>>;

	// Reference triangle
	Eigen::Vector3f ref_a{ 1, 0, 0 };
	Eigen::Vector3f ref_b{ 0, 1, 0 };
	Eigen::Vector3f ref_c{ 0, 0, 1 };

	vector3_t a{ 1, 0, 0 };
	vector3_t b{ 0, 1, 0 };
	vector3_t c{ 0, 0, 1 };

	size_t nr_problems = 80;

	std::vector<Eigen::Vector3f> ref_points(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points(nr_problems);

	// Strides
	size_t width = sizeof(real_t) / sizeof(float);

	// Results
	std::vector<float> d0(nr_problems);
	WideVector         d1(nr_problems / width);

	std::vector<float> s0(nr_problems);
	WideVector         s1(nr_problems / width);
	std::vector<float> t0(nr_problems);
	WideVector         t1(nr_problems / width);

	// Initialize data
	for (size_t i = 0; i < nr_problems; i++)
	{
		ref_points[i].setRandom();
		points.at<float>(i) = ref_points[i];
	}

	for (size_t i = 0; i < nr_problems; i++)
	{
		Eigen::Vector3f p = ref_points[i];
		std::array<float, 3> st;

		float d = distanceEberly(ref_a, ref_b, ref_c, p, &st);
		d0[i] = d;
		s0[i] = st[1];
		t0[i] = st[2];
	}
	for (size_t i = 0; i < nr_problems / width; i++)
	{
		vector3_t p = points.at<real_t>(i);
		std::array<real_t, 3> st;

		real_t d = distance({ a, b, c }, p, &st);
		d1[i] = d;
		s1[i] = st[1];
		t1[i] = st[2];
	}
	for (size_t i = 0; i < nr_problems; i++)
	{
		EXPECT_TRUE(equal(d0[i], reinterpret_cast<float*>(d1.data())[i], 1e-4f)) << "Distance differ: " << i;
		EXPECT_TRUE(equal(s0[i], reinterpret_cast<float*>(s1.data())[i], 1e-4f)) << "S differ: " << i;
		EXPECT_TRUE(equal(t0[i], reinterpret_cast<float*>(t1.data())[i], 1e-4f)) << "T differ: " << i;
	}
}

TEST(TriangleTriangleDistance, Simple)
{
	using namespace Vcl::Geometry;
	using Vcl::Mathematics::equal;

	//using real_t = Vcl::float16;
	//using real_t = Vcl::float8;
	using real_t = Vcl::float4;
	//using real_t = float;

	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	const size_t width = sizeof(real_t) / sizeof(float);
	const size_t problem_size = 64;

	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_a(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_b(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_c(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_A(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_B(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_C(problem_size);

	for (size_t i = 0; i < problem_size; i++)
	{
		points_a.at<float>(i) = Eigen::Vector3f::Random();
		points_b.at<float>(i) = Eigen::Vector3f::Random();
		points_c.at<float>(i) = Eigen::Vector3f::Random();
		points_A.at<float>(i) = Eigen::Vector3f::Random();
		points_B.at<float>(i) = Eigen::Vector3f::Random();
		points_C.at<float>(i) = Eigen::Vector3f::Random();
	}

	// Shortest distance
	Vcl::Core::InterleavedArray<float, 1, 1, -1> ref_shortest_dist(problem_size);
	Vcl::Core::InterleavedArray<float, 1, 1, -1> shortest_dist(problem_size);
	
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_x(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_y(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_X(problem_size);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points_Y(problem_size);

	// Compute the reference solution
	gte::DCPQuery<float, gte::Triangle3<float>, gte::Triangle3<float>> gteQuery;

	for (size_t i = 0; i < problem_size; i++)
	{
		Eigen::Vector3f triA_0 = points_a.at<float>(i);
		Eigen::Vector3f triA_1 = points_b.at<float>(i);
		Eigen::Vector3f triA_2 = points_c.at<float>(i);
		Eigen::Vector3f triB_0 = points_A.at<float>(i);
		Eigen::Vector3f triB_1 = points_B.at<float>(i);
		Eigen::Vector3f triB_2 = points_C.at<float>(i);

		gte::Triangle3<float> A{ cast(triA_0), cast(triA_1), cast(triA_2) };
		gte::Triangle3<float> B{ cast(triB_0), cast(triB_1), cast(triB_2) };

		auto res = gteQuery(A, B);
		ref_shortest_dist.at<float>(i)[0] = res.sqrDistance;
		points_x.at<float>(i) = cast(res.closestPoint[0]);
		points_y.at<float>(i) = cast(res.closestPoint[1]);
	}

	// Compute VCL solution
	for (size_t i = 0; i < problem_size / width; i++)
	{
		vector3_t triA_0 = points_a.at<real_t>(i);
		vector3_t triA_1 = points_b.at<real_t>(i);
		vector3_t triA_2 = points_c.at<real_t>(i);
		vector3_t triB_0 = points_A.at<real_t>(i);
		vector3_t triB_1 = points_B.at<real_t>(i);
		vector3_t triB_2 = points_C.at<real_t>(i);

		vector3_t a, b;
		real_t dist = distance({ triA_0, triA_1, triA_2 }, { triB_0, triB_1, triB_2 }, a, b);
		shortest_dist.at<real_t>(i)[0] = dist;
		points_X.at<real_t>(i) = a;
		points_Y.at<real_t>(i) = b;
	}

	for (size_t i = 0; i < problem_size; i++)
	{
		EXPECT_TRUE(equal(ref_shortest_dist.at<float>(i)[0], shortest_dist.at<float>(i)[0], 1e-4f)) << "Distance differ: " << i;
	}
}

TEST(DistanceRayRay, Simple)
{
	using Vcl::Geometry::distance;
	using Vcl::Geometry::Ray;
	using Vcl::Geometry::Result;
	using Vcl::Mathematics::equal;

	//using real_t = Vcl::float16;
	//using real_t = Vcl::float8;
	//using real_t = Vcl::float4;
	using real_t = float;

	// Simple crossing, intersection
	{
		Ray<real_t, 3> ray_a{ { -1, 0, 0 },{ 1, 0, 0 } };
		Ray<real_t, 3> ray_b{ { 0, -1, 0 },{ 0, 1, 0 } };

		Result<real_t> result;
		EXPECT_FLOAT_EQ(distance(ray_a, ray_b, &result), 0.0f);

		EXPECT_FLOAT_EQ(result.Parameter[0], 1.0f);
		EXPECT_FLOAT_EQ(result.Parameter[1], 1.0f);
	}

	// Simple crossing, no intersection
	{
		Ray<real_t, 3> ray_a{ { -1, 0, 0 },{ 1, 0, 0 } };
		Ray<real_t, 3> ray_b{ { 0, -1, 1 },{ 0, 1, 0 } };

		Result<real_t> result;
		EXPECT_FLOAT_EQ(distance(ray_a, ray_b, &result), 1.0f);

		EXPECT_FLOAT_EQ(result.Parameter[0], 1.0f);
		EXPECT_FLOAT_EQ(result.Parameter[1], 1.0f);
	}

	// Parallel crossing, intersection
	{
		Ray<real_t, 3> ray_a{ { -1, 0, 0 },{ 1, 0, 0 } };
		Ray<real_t, 3> ray_b{ { -1, 0, 0 },{ 1, 0, 0 } };

		Result<real_t> result;
		EXPECT_FLOAT_EQ(distance(ray_a, ray_b, &result), 0.0f);

		EXPECT_FLOAT_EQ(result.Parameter[0], 0.0f);
		EXPECT_FLOAT_EQ(result.Parameter[1], 0.0f);
	}

	// Parallel crossing, no intersection
	{
		Ray<real_t, 3> ray_a{ { -1, 0, 0 },{ 1, 0, 0 } };
		Ray<real_t, 3> ray_b{ { -1, 0, 1 },{ 1, 0, 0 } };

		Result<real_t> result;
		EXPECT_FLOAT_EQ(distance(ray_a, ray_b, &result), 1.0f);

		EXPECT_FLOAT_EQ(result.Parameter[0], 0.0f);
		EXPECT_FLOAT_EQ(result.Parameter[1], 0.0f);
	}
}
