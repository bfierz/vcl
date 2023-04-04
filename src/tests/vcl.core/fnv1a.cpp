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
#include <vcl/util/hashedstring.h>

VCL_BEGIN_EXTERNAL_HEADERS

// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

using namespace Vcl::Util;
using namespace Vcl::Util::Literals;

TEST(Fnv1aTest, 32bit)
{
	EXPECT_EQ(StringHash("Hello, World!").hash(), 0x5aecf734);
	EXPECT_EQ(calculateFnv1a32("Hello, World!"), 0x5aecf734);
	EXPECT_EQ("Hello, World!"_fnv1a32, 0x5aecf734);
}

TEST(Fnv1aTest, 64bit)
{
	EXPECT_EQ(calculateFnv1a64("Hello, World!"), 0x6ef05bd7cc857c54);
	EXPECT_EQ("Hello, World!"_fnv1a64, 0x6ef05bd7cc857c54);
}
