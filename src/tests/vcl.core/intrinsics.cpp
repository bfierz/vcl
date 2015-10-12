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

// Include the relevant parts from the library
#include <vcl/math/math.h>

VCL_BEGIN_EXTERNAL_HEADERS

// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

using Vcl::Mathematics::clz16;
using Vcl::Mathematics::clz32;
using Vcl::Mathematics::clz64;

TEST(Clz, 16bit)
{
	EXPECT_EQ(clz16(0), 16);
	EXPECT_EQ(clz16(1 << 0), 15);
	EXPECT_EQ(clz16(1 << 9), 6);
	EXPECT_EQ(clz16(1 << 15), 0);
}

TEST(Clz, 32bit)
{
	EXPECT_EQ(clz32(0), 32);
	EXPECT_EQ(clz32(1 << 0), 31);
	EXPECT_EQ(clz32(1 << 9), 22);
	EXPECT_EQ(clz32(1 << 31), 0);
}

TEST(Clz, 64bit)
{
	EXPECT_EQ(clz64(0ull), 64);
	EXPECT_EQ(clz64(1ull << 0), 63);
	EXPECT_EQ(clz64(1ull << 9), 54);
	EXPECT_EQ(clz64(1ull << 63), 0);
}
