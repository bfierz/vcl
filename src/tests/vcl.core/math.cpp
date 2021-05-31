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

// C++ standard library
#include <random>

// Include the relevant parts from the library
#include <vcl/math/math.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

using namespace Vcl::Mathematics;

TEST(Math, FltSign)
{
	std::random_device rnd;
	std::uniform_real_distribution<double> dist_d{-10, 10};

	EXPECT_EQ(sgn(0.0), 0.0);
	EXPECT_EQ(sgn(-0.0), 0.0);
	for (int i = 0; i < 50; i++)
	{
		const double d = dist_d(rnd);
		if (d < 0)
		{
			EXPECT_EQ(sgn(d), -1.0);
		}
		else if (d > 0)
		{
			EXPECT_EQ(sgn(d), 1.0);
		}
	}
}

TEST(Math, IntSign)
{
	std::random_device rnd;
	std::uniform_int_distribution<int> dist{ -10, 10 };

	EXPECT_EQ(sgn(0), 0);
	for (int i = 0; i < 50; i++)
	{
		const int d = dist(rnd);
		if (d < 0)
		{
			EXPECT_EQ(sgn(d), -1);
		}
		else if (d > 0)
		{
			EXPECT_EQ(sgn(d), 1);
		}
	}
}

TEST(Math, Max)
{
	std::random_device rnd;
	std::uniform_real_distribution<double> dist_d{ -10, 10 };

	for (int i = 0; i < 50; i++)
	{
		const double d0 = dist_d(rnd);
		const double d1 = dist_d(rnd);
		if (d0 > d1)
		{
			EXPECT_EQ(max(d0, d1), d0);
		}
		else
		{
			EXPECT_EQ(max(d0, d1), d1);
		}
	}
}

TEST(Math, Min)
{
	std::random_device rnd;
	std::uniform_real_distribution<double> dist_d{ -10, 10 };

	for (int i = 0; i < 50; i++)
	{
		const double d0 = dist_d(rnd);
		const double d1 = dist_d(rnd);
		if (d0 < d1)
		{
			EXPECT_EQ(min(d0, d1), d0);
		}
		else
		{
			EXPECT_EQ(min(d0, d1), d1);
		}
	}
}
