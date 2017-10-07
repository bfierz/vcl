/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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

// C++ standard library
#include <cmath>
#include <sstream>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

#if defined(VCL_COMPILER_MSVC)
#pragma push
#pragma float_control(precise, on)
#endif
template<class T>
const T& min(const T& a, const T& b)
{
	return (a < b) ? a : b;
}

template<class T>
const T& max(const T& a, const T& b)
{
	return (a > b) ? a : b;
}
#if defined(VCL_COMPILER_MSVC)
#pragma pop
#endif

TEST(MinMax, NanSafeMinMax)
{
	// Prepare a buffer with a NaN float
	float in_nan = std::nanf("");
	std::stringstream nan_stream(std::ios_base::in | std::ios_base::out | std::ios_base::binary);
	nan_stream.write((char*) &in_nan, sizeof(in_nan));
	nan_stream.seekg(0);

	float nan = 0;
	nan_stream.read((char*)&nan, sizeof(nan));

	// Do the tests
	EXPECT_TRUE(std::isnan(min(0.0f, nan)));
	EXPECT_TRUE(min(nan, 0.0f) == 0.0f);
	EXPECT_TRUE(std::isnan(max(0.0f, nan)));
	EXPECT_TRUE(max(nan, 0.0f) == 0.0f);

	EXPECT_TRUE(std::isnan(Vcl::Mathematics::min(0.0f, nan)));
	EXPECT_TRUE(Vcl::Mathematics::min(nan, 0.0f) == 0.0f);
	EXPECT_TRUE(std::isnan(Vcl::Mathematics::max(0.0f, nan)));
	EXPECT_TRUE(Vcl::Mathematics::max(nan, 0.0f) == 0.0f);
}
