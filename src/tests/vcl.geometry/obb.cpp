/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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

// Include the relevant parts from the library
#include <vcl/geometry/primitives/obb.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>

// Local
#include "liver_766.h"
VCL_END_EXTERNAL_HEADERS

TEST(OrientedBoxTest, SimpleConstruction)
{
	using namespace Vcl::Geometry;
	
	// Basic box
	Eigen::Vector3f min_pos = Eigen::Vector3f::Constant(-1);
	Eigen::Vector3f max_pos = Eigen::Vector3f::Constant( 1);
	Eigen::AlignedBox3f aligned_box{ min_pos, max_pos };
	
	// Define random rotation
	Eigen::AngleAxisf rot{ 0.785f, Eigen::Vector3f::Random().normalized() };

	size_t c = 0;
	std::vector<Eigen::Vector3f> points;
	std::generate_n(std::back_inserter(points), 8, [&aligned_box, &rot, &c]()
	{
		return rot * aligned_box.corner(static_cast<Eigen::AlignedBox3f::CornerType>(c++));
	});
	
	OrientedBox<float, 3> obb{ points };

	EXPECT_FLOAT_EQ(obb.center().x(), 0);
	EXPECT_FLOAT_EQ(obb.center().y(), 0);
	EXPECT_FLOAT_EQ(obb.center().z(), 0);
}

TEST(OrientedBoxTest, ConstructionFromMesh)
{
	using namespace Vcl::Geometry;

	OrientedBox<float, 3> obb{ { reinterpret_cast<const Eigen::Vector3f*>(liver_766_points), num_liver_766_points / 3 } };
}
