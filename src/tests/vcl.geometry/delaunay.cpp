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
#include <random>

// Include the relevant parts from the library
#include <vcl/geometry/delaunay.h>

// Google test
#include <gtest/gtest.h>

TEST(DelaunayTest, SimpleExample)
{
	using namespace Vcl::Geometry;

	std::vector<Eigen::Vector2f> points(10);
	points[0] = { 0, 1 };
	points[1] = { 1, 0 };
	points[2] = { 1, 1 };
	points[3] = { 1, 2 };
	points[4] = { 2, 0 };
	points[5] = { 3, 3 };
	points[6] = { 4, 2 };
	points[7] = { 5, 0 };
	points[8] = { 5, 1 };
	points[9] = { 5, 3 };

	auto mesh = computeDelaunayTriangulation(points);
	EXPECT_EQ(mesh.nrVertices(), 10);
}

TEST(DelaunayTest, GridConstruction)
{
	using namespace Vcl::Geometry;

	constexpr size_t side = 3;

	std::vector<Eigen::Vector2f> points;
	points.reserve(side*side);

	for (size_t y = 0; y < side; y++)
	{
		for (size_t x = 0; x < side; x++)
		{
			points.emplace_back(x, y);
		}
	}

	// Randomize the points in order to test the sorting
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(std::begin(points), std::end(points), g);

	auto mesh = computeDelaunayTriangulation(points);
	EXPECT_EQ(mesh.nrVertices(), side*side);
}
