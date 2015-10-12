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

// Include the relevant parts from the library
#include <vcl/core/container/octree.h>

// C++ standard library
#include <random>
#include <vector>

// Google test
#include <gtest/gtest.h>

namespace
{
	const std::vector<Eigen::Vector3f> pointsModel =
	{
	#include "points.h"
	};
	const Eigen::AlignedBox3f boxModel = []()
	{
		Eigen::AlignedBox3f box;
		for (auto& p : pointsModel)
			box.extend(p);
		return box;
	}();

	const std::vector<Eigen::Vector3f> points4x4x4 = []()
	{
		std::vector<Eigen::Vector3f> points;
		for (int k = 0; k < 4; k++)
		for (int j = 0; j < 4; j++)
		for (int i = 0; i < 4; i++)
		{
			points.emplace_back(float(i) + 0.25f, float(j) + 0.25f, float(k) + 0.25f);
		}
		return points;
	}();
	const Eigen::AlignedBox3f box4x4x4(Eigen::Vector3f::Constant(0), Eigen::Vector3f::Constant(4));

	const std::vector<Eigen::Vector3f> points8x8x8 = []()
	{
		std::vector<Eigen::Vector3f> points;
		for (int k = 0; k < 8; k++)
		for (int j = 0; j < 8; j++)
		for (int i = 0; i < 8; i++)
		{
			points.emplace_back(float(i) + 0.25f, float(j) + 0.25f, float(k) + 0.25f);
		}
		return points;
	}();
	const Eigen::AlignedBox3f box8x8x8(Eigen::Vector3f::Constant(0), Eigen::Vector3f::Constant(8));

	struct OctreeTestParams
	{
		std::vector<Eigen::Vector3f> points;
		Eigen::AlignedBox3f box;
		int depth;
	};
	OctreeTestParams testParamsModel = { pointsModel, boxModel, 5 };
	OctreeTestParams testParams4x4x4 = { points4x4x4, box4x4x4, 3 };
	OctreeTestParams testParams8x8x8 = { points8x8x8, box8x8x8, 4 };
}

class OctreeTest : public ::testing::TestWithParam<OctreeTestParams>
{
};

using namespace Vcl::Core;

TEST_P(OctreeTest, FindBox)
{
	using namespace Vcl;

	const auto& points = GetParam().points;
	const auto& box = GetParam().box;
	const auto& depth = GetParam().depth;

	Octree tree(box, depth);
	tree.assign(points);

	const Eigen::Vector3f search_point = points[20];
	const float search_radius = 2.0f;
	const Eigen::AlignedBox3f search_box
	{
		search_point - Eigen::Vector3f::Constant(search_radius),
		search_point + Eigen::Vector3f::Constant(search_radius)
	};

	std::vector<int> found_points;
	tree.find(search_box, found_points);

	std::vector<int> ref_found_points;
	ref_found_points.reserve(found_points.size());

	for (size_t i = 0; i < points.size(); i++)
	{
		const auto& p = points[i];
		if (search_box.contains(p))
		{
			ref_found_points.emplace_back(i);
		}
	}

	std::sort(std::begin(found_points), std::end(found_points));
	std::sort(std::begin(ref_found_points), std::end(ref_found_points));

	EXPECT_EQ(found_points, ref_found_points);
}
TEST_P(OctreeTest, FindRadius)
{
	using namespace Vcl;

	const auto& points = GetParam().points;
	const auto& box = GetParam().box;
	const auto& depth = GetParam().depth;

	Octree tree(box, depth);
	tree.assign(points);

	const Eigen::Vector3f search_point = points[20];
	const float search_radius = 2.0f;

	std::vector<int> found_points;
	tree.find(search_point, search_radius, found_points);

	std::vector<int> ref_found_points;
	ref_found_points.reserve(found_points.size());

	for (size_t i = 0; i < points.size(); i++)
	{
		const auto& p = points[i];
		if ((search_point - p).squaredNorm() <= search_radius*search_radius)
		{
			ref_found_points.emplace_back(i);
		}
	}

	std::sort(std::begin(found_points), std::end(found_points));
	std::sort(std::begin(ref_found_points), std::end(ref_found_points));

	EXPECT_EQ(found_points, ref_found_points);
}
INSTANTIATE_TEST_CASE_P(Octree,
	OctreeTest,
	::testing::Values(testParams4x4x4, testParams8x8x8, testParamsModel));
