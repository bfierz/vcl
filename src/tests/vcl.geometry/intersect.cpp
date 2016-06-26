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
#include <vcl/geometry/intersect.h>

// Google test
#include <gtest/gtest.h>

template<typename Scalar, typename Func>
void testAxisAlignedIntersection(const Vcl::Geometry::Ray<Scalar, 3>& ray, Func intersect, bool result)
{
	using Vcl::Mathematics::equal;
	using Vcl::all;

	using real_t = Scalar;

	using vec3_t = Eigen::Matrix<real_t, 3, 1>;
	using box3_t = Eigen::AlignedBox<real_t, 3>;

	box3_t b0{ vec3_t{ 0, 0, 0 }, vec3_t{ 1, 1, 1 } };

	EXPECT_EQ(result, all(intersect(b0, ray))) << "Intersection was missed.";
}

TEST(AxisAlignedBoxRayIntersection, ScalarBarnes)
{
	Vcl::Geometry::Ray<float, 3> r0{ { 0.5f, 0.5f, 0.0f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r1{ { 0.0f, 0.0f, 0.0f },{ 0, 0, 1 } };

	Vcl::Geometry::Ray<float, 3> r2{ { 1.0f, 0.0f, -0.000001f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r3{ { 1.0f, 0.0f,  0.0f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r4{ { 1.0f, 0.0f,  1.000001f },{ 0, 0, 1 } };

	typedef bool (*Func) (const Eigen::AlignedBox<float, 3>&, const Vcl::Geometry::Ray<float, 3>&);
	Func f = Vcl::Geometry::intersects;
	testAxisAlignedIntersection(r0, f, true);
	testAxisAlignedIntersection(r1, f, true);
	testAxisAlignedIntersection(r2, f, true);
	testAxisAlignedIntersection(r3, f, true);
	testAxisAlignedIntersection(r4, f, false);
}

TEST(AxisAlignedBoxRayIntersection, ScalarIze)
{
	Vcl::Geometry::Ray<float, 3> r0{ { 0.5f, 0.5f, 0.0f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r1{ { 0.0f, 0.0f, 0.0f },{ 0, 0, 1 } };

	Vcl::Geometry::Ray<float, 3> r2{ { 1.0f, 0.0f, -0.000001f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r3{ { 1.0f, 0.0f,  0.0f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r4{ { 1.0f, 0.0f,  1.000001f },{ 0, 0, 1 } };

	typedef bool(*Func) (const Eigen::AlignedBox<float, 3>&, const Vcl::Geometry::Ray<float, 3>&);
	Func f = Vcl::Geometry::intersects_MaxMult;
	testAxisAlignedIntersection(r0, f, true);
	testAxisAlignedIntersection(r1, f, true);
	testAxisAlignedIntersection(r2, f, true);
	testAxisAlignedIntersection(r3, f, true);
	testAxisAlignedIntersection(r4, f, false);
}

TEST(AxisAlignedBoxRayIntersection, ScalarPharr)
{
	Vcl::Geometry::Ray<float, 3> r0{ { 0.5f, 0.5f, 0.0f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r1{ { 0.0f, 0.0f, 0.0f },{ 0, 0, 1 } };

	Vcl::Geometry::Ray<float, 3> r2{ { 1.0f, 0.0f, -0.000001f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r3{ { 1.0f, 0.0f,  0.0f },{ 0, 0, 1 } };
	Vcl::Geometry::Ray<float, 3> r4{ { 1.0f, 0.0f,  1.000001f },{ 0, 0, 1 } };

	typedef bool(*Func) (const Eigen::AlignedBox<float, 3>&, const Vcl::Geometry::Ray<float, 3>&);
	Func f = Vcl::Geometry::intersects_Pharr;
	testAxisAlignedIntersection(r0, f, true);
	testAxisAlignedIntersection(r1, f, true);
	testAxisAlignedIntersection(r2, f, true);
	testAxisAlignedIntersection(r3, f, true);
	testAxisAlignedIntersection(r4, f, false);
}

TEST(AxisAlignedBoxRayIntersection, SimpleFloat)
{
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;
	using Vcl::all;

	using real_t = float;

	using vec3_t = Eigen::Matrix<real_t, 3, 1>;
	using box3_t = Eigen::AlignedBox<real_t, 3>;
	using ray3_t = Vcl::Geometry::Ray<real_t, 3>;

	box3_t b0{ vec3_t{ 0, 0, 0 }, vec3_t{ 1, 1, 1 } };
	ray3_t r{ { 2.0f, 2.0f, 0.0f }, { 0, 0, -1 } };

	EXPECT_TRUE(all(intersects(b0, r))) << "Intersection was missed.";
}

TEST(AxisAlignedBoxRayIntersection, SimpleFloat4)
{
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;
	using Vcl::all;

	using real_t = Vcl::float4;

	using vec3_t = Eigen::Matrix<real_t, 3, 1>;
	using box3_t = Eigen::AlignedBox<real_t, 3>;
	using ray3_t = Vcl::Geometry::Ray<real_t, 3>;

	box3_t b0{ vec3_t{ 0, 0, 0 }, vec3_t{ 1, 1, 1 } };
	ray3_t r{ { 2.0f, 2.0f, 0.0f }, { 0, 0, -1 } };

	EXPECT_TRUE(all(intersects(b0, r))) << "Intersection was missed.";
}

TEST(AxisAlignedBoxRayIntersection, SimpleFloat8)
{
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;
	using Vcl::all;

	using real_t = Vcl::float8;

	using vec3_t = Eigen::Matrix<real_t, 3, 1>;
	using box3_t = Eigen::AlignedBox<real_t, 3>;
	using ray3_t = Vcl::Geometry::Ray<real_t, 3>;

	box3_t b0{ vec3_t{ 0, 0, 0 }, vec3_t{ 1, 1, 1 } };
	ray3_t r{ { 2.0f, 2.0f, 0.0f }, { 0, 0, -1 } };

	EXPECT_TRUE(all(intersects(b0, r))) << "Intersection was missed.";
}

TEST(AxisAlignedBoxRayIntersection, SimpleFloat16)
{
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;
	using Vcl::all;

	using real_t = Vcl::float16;

	using vec3_t = Eigen::Matrix<real_t, 3, 1>;
	using box3_t = Eigen::AlignedBox<real_t, 3>;
	using ray3_t = Vcl::Geometry::Ray<real_t, 3>;

	box3_t b0{ vec3_t{ 0, 0, 0 }, vec3_t{ 1, 1, 1 } };
	ray3_t r{ { 2.0f, 2.0f, 0.0f }, { 0, 0, -1 } };

	EXPECT_TRUE(all(intersects(b0, r))) << "Intersection was missed.";
}