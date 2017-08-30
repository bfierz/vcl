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
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/math/math.h>

// C++ standard library
#include <cmath>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(Simd, Rsqrt)
{
	using Vcl::float4;

	using Vcl::all;
	using Vcl::rsqrt;

	using Vcl::Mathematics::equal;

	// Source data
	float4 vec{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute 1 / sqrt(x)
	float4 res = rsqrt(vec);

	// Reference result
	float4 ref{ 0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f };

	EXPECT_TRUE(all(equal(ref, res, float4(1e-5f)))) << "'rsqrt' failed.";
}

TEST(Simd, Rcp)
{
	using Vcl::float4;

	using Vcl::all;
	using Vcl::rsqrt;

	using Vcl::Mathematics::equal;

	// Source data
	float4 vec{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute 1 / x
	float4 res = rcp(vec);

	// Reference result
	float4 ref{ 0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f };

	EXPECT_TRUE(all(equal(ref, res, float4(1e-5f)))) << "'rsqrt' failed.";
}
