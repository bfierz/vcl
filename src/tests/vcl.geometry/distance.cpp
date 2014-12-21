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
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/geometry/distance.h>

#include "distance_ref.h"

// Google test
#include <gtest/gtest.h>

// Tests the scalar gather function.
TEST(PointTriangleDistance, Simple)
{
	using Vcl::Geometry::distance;
	using Vcl::Mathematics::equal;

	//typedef Vcl::float16 real_t;
	//typedef Vcl::float8 real_t;
	//typedef Vcl::float4 real_t;
	typedef float real_t;

	typedef Eigen::Matrix<real_t, 3, 1> vector3_t;

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
	std::vector<float>  d0(nr_problems);
	std::vector<real_t> d1(nr_problems / width);

	std::vector<float>  s0(nr_problems);
	std::vector<real_t> s1(nr_problems / width);
	std::vector<float>  t0(nr_problems);
	std::vector<real_t> t1(nr_problems / width);

	// Initialize data
	for (int i = 0; i < (int) nr_problems; i++)
	{
		ref_points[i].setRandom();
		points.at<float>(i) = ref_points[i];
	}

	for (int i = 0; i < nr_problems; i++)
	{
		Eigen::Vector3f p = ref_points[i];
		std::array<float, 3> st;

		float d = distanceEberly(ref_a, ref_b, ref_c, p, &st);
		d0[i] = d;
		s0[i] = st[1];
		t0[i] = st[2];
	}
	for (int i = 0; i < static_cast<int>(nr_problems / width); i++)
	{
		vector3_t p = points.at<real_t>(i);
		std::array<real_t, 3> st;

		real_t d = distance(a, b, c, p, &st);
		d1[i] = d;
		s1[i] = st[1];
		t1[i] = st[2];
	}
	for (int i = 0; i < nr_problems; i++)
	{
		EXPECT_TRUE(equal(d0[i], reinterpret_cast<float*>(d1.data())[i], 1e-3f)) << "Distance differ: " << i;
		EXPECT_TRUE(equal(s0[i], reinterpret_cast<float*>(s1.data())[i], 1e-3f)) << "S differ: " << i;
		EXPECT_TRUE(equal(t0[i], reinterpret_cast<float*>(t1.data())[i], 1e-3f)) << "T differ: " << i;
	}
}
