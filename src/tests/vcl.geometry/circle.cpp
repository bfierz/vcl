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

// Include the relevant parts from the library
#include <vcl/geometry/circle.h>

// Google test
#include <gtest/gtest.h>

// Tests the circle function
TEST(CircleTest, SimpleConstruction)
{
	using namespace Vcl::Geometry;

	{
		Circle<float, 2> circle{ {2,1}, {0,5}, {-1,2} };
		EXPECT_FLOAT_EQ(circle.center().x(), 1);
		EXPECT_FLOAT_EQ(circle.center().y(), 3);
		EXPECT_FLOAT_EQ(circle.radius()*circle.radius(), 5);
	}
	{
		Circle<float, 2> circle{ { 0,5 },{ 2,1 },{ -1,2 } };
		EXPECT_FLOAT_EQ(circle.center().x(), 1);
		EXPECT_FLOAT_EQ(circle.center().y(), 3);
		EXPECT_FLOAT_EQ(circle.radius()*circle.radius(), 5);
	}
	{
		Circle<float, 2> circle{ { -6, 5 },{ -3, -4 },{ 2, 1 } };
		EXPECT_FLOAT_EQ(circle.center().x(), -3);
		EXPECT_FLOAT_EQ(circle.center().y(), 1);
		EXPECT_FLOAT_EQ(circle.radius()*circle.radius(), 25);
	}
}

TEST(CircleTest, InOut)
{
	using namespace Vcl::Geometry;

	Eigen::Vector2f p0{ -6,  5 };
	Eigen::Vector2f p1{ -3, -4 };
	Eigen::Vector2f p2{  2,  1 };

	Eigen::Vector2f p;

	p = { -3, 1 };
	EXPECT_EQ(-1, isInCircle(p0, p1, p2, p));
	p = { 1.99f, 1 };
	EXPECT_EQ(-1, isInCircle(p0, p1, p2, p));
	p = { 2, 1 };
	EXPECT_EQ(0, isInCircle(p0, p1, p2, p));
	p = { 2.01f, 1 };
	EXPECT_EQ(1, isInCircle(p0, p1, p2, p));
	p = { 7, 1 };
	EXPECT_EQ(1, isInCircle(p0, p1, p2, p));
}
