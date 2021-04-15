/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2021 Basil Fierz
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
#include <array>
#include <random>
#include <vector>

// Include the relevant parts from the library
#include <vcl/core/container/array.h>
#include <vcl/geometry/intersect.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

namespace
{
	std::array<Eigen::Vector3f, 8> positions =
	{
		Eigen::Vector3f(0, 0, 0),
		Eigen::Vector3f(1, 0, 0),
		Eigen::Vector3f(0, 1, 0),
		Eigen::Vector3f(1, 1, 0),
		Eigen::Vector3f(0, 0, 1),
		Eigen::Vector3f(1, 0, 1),
		Eigen::Vector3f(0, 1, 1),
		Eigen::Vector3f(1, 1, 1),
	};
	std::array<std::array<int, 4>, 5> tetrahedra =
	{
		std::make_array(0, 5, 3, 6),
		std::make_array(0, 1, 3, 5),
		std::make_array(0, 2, 6, 3),
		std::make_array(0, 6, 4, 5),
		std::make_array(3, 6, 5, 7),
	};
}

TEST(RayTetIntersection, IntersectCenter)
{
	using namespace Vcl::Geometry;
	using Vcl::Mathematics::equal;
	
	using ray3_t = Vcl::Geometry::Ray<float, 3>;

	for (const auto tetrahedron : tetrahedra)
	{
		const Tetrahedron<float, 3> tet
		{
			positions[tetrahedron[0]],
			positions[tetrahedron[1]],
			positions[tetrahedron[2]],
			positions[tetrahedron[3]],
		};
		EXPECT_GT(tet.computeSignedVolume(), 0);
		const auto center = tet.computeCenter();

		// Intersection points
		const auto& a = tet[0];
		const auto& b = tet[1];
		const auto& c = tet[2];
		const auto& d = tet[3];

		const ray3_t ray_face_abc{ (center + 2 * (center - d)).eval(), (d - center).normalized() };
		const ray3_t ray_face_abd{ (center + 2 * (center - c)).eval(), (c - center).normalized() };
		const ray3_t ray_face_acd{ (center + 2 * (center - b)).eval(), (b - center).normalized() };
		const ray3_t ray_face_bcd{ (center + 2 * (center - a)).eval(), (a - center).normalized() };

		EXPECT_TRUE(intersects(tet, ray_face_abc));
		EXPECT_TRUE(intersects(tet, ray_face_abd));
		EXPECT_TRUE(intersects(tet, ray_face_acd));
		EXPECT_TRUE(intersects(tet, ray_face_bcd));
	}
}

TEST(RayTetIntersection, NoIntersectionParallelRay)
{
	using namespace Vcl::Geometry;
	using Vcl::Mathematics::equal;

	using ray3_t = Vcl::Geometry::Ray<float, 3>;

	for (const auto tetrahedron : tetrahedra)
	{
		const Tetrahedron<float, 3> tet
		{
			positions[tetrahedron[0]],
			positions[tetrahedron[1]],
			positions[tetrahedron[2]],
			positions[tetrahedron[3]],
		};
		EXPECT_GT(tet.computeSignedVolume(), 0);
		const auto center = tet.computeCenter();

		// Intersection points
		const auto& a = tet[0];
		const auto& b = tet[1];
		const auto& c = tet[2];
		const auto& d = tet[3];

		const ray3_t ray_face_abc{ (center + 2 * (center - d)).eval(), (b - a).normalized() };
		const ray3_t ray_face_abd{ (center + 2 * (center - c)).eval(), (d - a).normalized() };
		const ray3_t ray_face_acd{ (center + 2 * (center - b)).eval(), (d - a).normalized() };
		const ray3_t ray_face_bcd{ (center + 2 * (center - a)).eval(), (d - b).normalized() };

		EXPECT_FALSE(intersects(tet, ray_face_abc));
		EXPECT_FALSE(intersects(tet, ray_face_abd));
		EXPECT_FALSE(intersects(tet, ray_face_acd));
		EXPECT_FALSE(intersects(tet, ray_face_bcd));
	}
}

TEST(RayTetIntersection, IntersectFacePlaneNotTet)
{
	using namespace Vcl::Geometry;
	using Vcl::Mathematics::equal;

	using ray3_t = Vcl::Geometry::Ray<float, 3>;

	for (const auto tetrahedron : tetrahedra)
	{
		const Tetrahedron<float, 3> tet
		{
			positions[tetrahedron[0]],
			positions[tetrahedron[1]],
			positions[tetrahedron[2]],
			positions[tetrahedron[3]],
		};
		EXPECT_GT(tet.computeSignedVolume(), 0);
		const auto center = tet.computeCenter();

		// Intersection points
		const auto& a = tet[0];
		const auto& b = tet[1];
		const auto& c = tet[2];
		const auto& d = tet[3];

		const auto center_abc = ((a + b + c) / 3.0f).eval();
		const auto center_abd = ((a + b + d) / 3.0f).eval();
		const auto center_acd = ((a + c + d) / 3.0f).eval();
		const auto center_bcd = ((b + c + d) / 3.0f).eval();

		const ray3_t ray_face_abc{ (center + 2 * (center - d)).eval(), (center_abc - a).normalized() };
		const ray3_t ray_face_abd{ (center + 2 * (center - c)).eval(), (center_abd - a).normalized() };
		const ray3_t ray_face_acd{ (center + 2 * (center - b)).eval(), (center_acd - a).normalized() };
		const ray3_t ray_face_bcd{ (center + 2 * (center - a)).eval(), (center_bcd - b).normalized() };

		EXPECT_FALSE(intersects(tet, ray_face_abc));
		EXPECT_FALSE(intersects(tet, ray_face_abd));
		EXPECT_FALSE(intersects(tet, ray_face_acd));
		EXPECT_FALSE(intersects(tet, ray_face_bcd));
	}
}
