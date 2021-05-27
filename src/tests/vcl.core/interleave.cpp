/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2021 Basil Fierz
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
#include <vcl/core/simd/memory.h>

// C++ standard library
#include <cmath>
#include <sstream>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

#define VCL_SIMD_FLOATS \
	using float4 = Vcl::float4; \
	using float8 = Vcl::float8; \
	using float16 = Vcl::float16; \
	float4 f4_asc{1, 2, 3, 4}; \
	float4 f4_desc{4, 3, 2, 1}; \
	float8 f8_asc{1, 2, 3, 4, 5, 6, 7, 8}; \
	float8 f8_desc{8, 7, 6, 5, 4, 3, 2, 1}; \
	float16 f16_asc{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; \
	float16 f16_desc{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

TEST(Simd, Interleave)
{
	using Vcl::interleave;

	VCL_SIMD_FLOATS

	const auto f4 = interleave(f4_asc, f4_desc);
	for (int i = 0; i < 2; i++)
	{
		EXPECT_EQ(f4[0][2*i+0], f4_asc[i]);
		EXPECT_EQ(f4[0][2*i+1], f4_desc[i]);
		
		EXPECT_EQ(f4[1][2*i+0], f4_asc[i+2]);
		EXPECT_EQ(f4[1][2*i+1], f4_desc[i+2]);
	}

	const auto f8 = interleave(f8_asc, f8_desc);
	for (int i = 0; i < 4; i++)
	{
		EXPECT_EQ(f8[0][2 * i + 0], f8_asc[i]);
		EXPECT_EQ(f8[0][2 * i + 1], f8_desc[i]);

		EXPECT_EQ(f8[1][2 * i + 0], f8_asc[i + 4]);
		EXPECT_EQ(f8[1][2 * i + 1], f8_desc[i + 4]);
	}

	const auto f16 = interleave(f16_asc, f16_desc);
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(f16[0][2 * i + 0], f16_asc[i]);
		EXPECT_EQ(f16[0][2 * i + 1], f16_desc[i]);

		EXPECT_EQ(f16[1][2 * i + 0], f16_asc[i + 8]);
		EXPECT_EQ(f16[1][2 * i + 1], f16_desc[i + 8]);
	}
}
