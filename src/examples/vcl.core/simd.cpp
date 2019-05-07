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
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/math/math.h>

// C++ standard library
#include <cmath>
#include <random>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(CoreSimd, SimpleSingleOperation_F4)
{
//! [Add float4]
Vcl::float4 x{ 1.0f, 2.0f, 3.0f, 4.0f };
Vcl::float4 y{ 4.0f, 3.0f, 2.0f, 1.0f };

Vcl::float4 z = x + y;
//! [Add float4]

EXPECT_TRUE(Vcl::all(equal(z, Vcl::float4(5.0f), Vcl::float4(1e-5f))));
}

TEST(CoreSimd, SimpleSingleOperation_F8)
{
//! [Add float8]
Vcl::float8 x{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
Vcl::float8 y{ 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };

Vcl::float8 z = x + y;
//! [Add float8]

EXPECT_TRUE(Vcl::all(equal(z, Vcl::float8(9.0f), Vcl::float8(1e-5f))));
}
