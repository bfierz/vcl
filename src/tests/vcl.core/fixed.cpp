/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2019 Basil Fierz
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
#include <vcl/math/fixed.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

template<typename Fixed, typename Floating>
void success(Fixed result, Floating ref_val)
{
	EXPECT_EQ(result, Fixed{ref_val});
	EXPECT_FLOAT_EQ(static_cast<Floating>(result), ref_val);
}

template<typename Fixed, typename Floating>
void fail(Fixed result, Floating ref_val)
{
	using Vcl::Mathematics::equal;
	EXPECT_FALSE(equal(static_cast<Floating>(result), ref_val, 1e-2f));
}

TEST(FixedPointMathTest, Add_Should_Succeed_When_SumInRange)
{
	using namespace Vcl::Mathematics;

	fixed<short, 11> f1{1.0f};
	success(f1+f1, 2.0f);
	
	fixed<short, 11> f2{1.0f};
	f2 += f1;
	success(f2, 2.0f);
	
	fixed<short, 11> f15{1.5f};
	fixed<short, 11> f12{1.2f};
	success(f15+f12, static_cast<float>(f15)+static_cast<float>(f12));
	
	fixed<short, 11> f27{1.5f};
	f27 += f12;
	success(f27, static_cast<float>(f15)+static_cast<float>(f12));
}

TEST(FixedPointMathTest, Add_Should_Fail_When_SumOutOfRange)
{
	using namespace Vcl::Mathematics;

	fixed<short, 11> f1{8.0f};
	fail(f1+f1, 16.0f);
	
	fixed<short, 11> f15{15.5f};
	fixed<short, 11> f12{1.2f};
	fail(f15+f12, 16.7f);
}

TEST(FixedPointMathTest, Negate_Should_Succeed_When_Signed)
{
	using namespace Vcl::Mathematics;

	fixed<short, 11> f15{1.5f};
	success(-f15, -1.5f);
}

TEST(FixedPointMathTest, Negate_Should_Fail_When_Unsigned)
{
	using namespace Vcl::Mathematics;

	fixed<unsigned short, 11> f15{1.5f};
	fail(-f15, -1.5f);
}
