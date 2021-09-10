/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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

// Include the relevant parts from the library
#include <vcl/core/container/array.h>
#include <vcl/geometry/intersect.h>
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/tetramesh.h>

// Google test
#include <gtest/gtest.h>

std::array<Eigen::Vector3f, 8> large_positions = {
	Eigen::Vector3f(0, 0, 0),
	Eigen::Vector3f(1, 0, 0),
	Eigen::Vector3f(0, 1, 0),
	Eigen::Vector3f(1, 1, 0),
	Eigen::Vector3f(0, 0, 1),
	Eigen::Vector3f(1, 0, 1),
	Eigen::Vector3f(0, 1, 1),
	Eigen::Vector3f(1, 1, 1),
};
std::array<Eigen::Vector3f, 8> small_positions = {
	Eigen::Vector3f(0.25f, 0.25f, 0.25f),
	Eigen::Vector3f(0.75f, 0.25f, 0.25f),
	Eigen::Vector3f(0.25f, 0.75f, 0.25f),
	Eigen::Vector3f(0.75f, 0.75f, 0.25f),
	Eigen::Vector3f(0.25f, 0.25f, 0.75f),
	Eigen::Vector3f(0.75f, 0.25f, 0.75f),
	Eigen::Vector3f(0.25f, 0.75f, 0.75f),
	Eigen::Vector3f(0.75f, 0.75f, 0.75f),
};
std::array<std::array<int, 4>, 5> tetrahedra = {
	std::make_array(0, 5, 3, 6),
	std::make_array(0, 1, 3, 5),
	std::make_array(0, 2, 3, 6),
	std::make_array(0, 6, 5, 4),
	std::make_array(3, 6, 5, 7),
};

TEST(TetTetIntersection, Same)
{
	using namespace Vcl::Geometry;
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;

	Tetrahedron<float, 3> a{
		large_positions[tetrahedra[0][0]],
		large_positions[tetrahedra[0][1]],
		large_positions[tetrahedra[0][2]],
		large_positions[tetrahedra[0][3]],
	};

	Tetrahedron<float, 3> b{
		large_positions[tetrahedra[0][0]],
		large_positions[tetrahedra[0][1]],
		large_positions[tetrahedra[0][2]],
		large_positions[tetrahedra[0][3]],
	};

	EXPECT_TRUE(intersects(a, b));
}

TEST(TetTetIntersection, Touch)
{
	using namespace Vcl::Geometry;
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;

	Tetrahedron<float, 3> a{
		large_positions[tetrahedra[0][0]],
		large_positions[tetrahedra[0][1]],
		large_positions[tetrahedra[0][2]],
		large_positions[tetrahedra[0][3]],
	};

	Tetrahedron<float, 3> b{
		large_positions[tetrahedra[1][0]],
		large_positions[tetrahedra[1][1]],
		large_positions[tetrahedra[1][2]],
		large_positions[tetrahedra[1][3]],
	};

	EXPECT_TRUE(intersects(a, b));
}

TEST(TetTetIntersection, LargeIntersect)
{
	using namespace Vcl::Geometry;
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;

	Tetrahedron<float, 3> a{
		large_positions[tetrahedra[2][0]],
		large_positions[tetrahedra[2][1]],
		large_positions[tetrahedra[2][2]],
		large_positions[tetrahedra[2][3]],
	};

	Tetrahedron<float, 3> b{
		Eigen::Vector3f(0.0f, 0.5f, 0.0f) + large_positions[tetrahedra[3][0]],
		Eigen::Vector3f(0.0f, 0.5f, 0.0f) + large_positions[tetrahedra[3][1]],
		Eigen::Vector3f(0.0f, 0.5f, 0.0f) + large_positions[tetrahedra[3][2]],
		Eigen::Vector3f(0.0f, 0.5f, 0.0f) + large_positions[tetrahedra[3][3]],
	};

	EXPECT_TRUE(intersects(a, b));
}

TEST(TetTetIntersection, NonIntersect)
{
	using namespace Vcl::Geometry;
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;

	Tetrahedron<float, 3> a{
		large_positions[tetrahedra[0][0]],
		large_positions[tetrahedra[0][1]],
		large_positions[tetrahedra[0][2]],
		large_positions[tetrahedra[0][3]],
	};

	Tetrahedron<float, 3> b{
		Eigen::Vector3f(2, 0, 0) + large_positions[tetrahedra[0][0]],
		Eigen::Vector3f(2, 0, 0) + large_positions[tetrahedra[0][1]],
		Eigen::Vector3f(2, 0, 0) + large_positions[tetrahedra[0][2]],
		Eigen::Vector3f(2, 0, 0) + large_positions[tetrahedra[0][3]],
	};

	EXPECT_FALSE(intersects(a, b));
}

TEST(TetTetIntersection, Include)
{
	using namespace Vcl::Geometry;
	using Vcl::Geometry::intersects;
	using Vcl::Mathematics::equal;

	Tetrahedron<float, 3> a{
		large_positions[tetrahedra[0][0]],
		large_positions[tetrahedra[0][1]],
		large_positions[tetrahedra[0][2]],
		large_positions[tetrahedra[0][3]],
	};

	Tetrahedron<float, 3> b{
		small_positions[tetrahedra[0][0]],
		small_positions[tetrahedra[0][1]],
		small_positions[tetrahedra[0][2]],
		small_positions[tetrahedra[0][3]],
	};

	EXPECT_FALSE(intersects(a, b));
}
