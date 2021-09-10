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
#include <random>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

#define VCL_SIMD_BOOLS                                                  \
	using bool4 = Vcl::bool4;                                           \
	using bool8 = Vcl::bool8;                                           \
	using bool16 = Vcl::bool16;                                         \
	bool4 b4_0{ true, false, true, false };                             \
	bool4 b4_1{ false, true, false, true };                             \
	bool8 b8_0{ true, false, true, false, true, false, true, false };   \
	bool8 b8_1{ false, true, false, true, false, true, false, true };   \
	bool16 b16_0{ true, false, true, false, true, false, true, false,   \
				  true, false, true, false, true, false, true, false }; \
	bool16 b16_1{ false, true, false, true, false, true, false, true,   \
				  false, true, false, true, false, true, false, true };

#define VCL_SIMD_FLOATS                                                       \
	using float4 = Vcl::float4;                                               \
	using float8 = Vcl::float8;                                               \
	using float16 = Vcl::float16;                                             \
	float4 f4_asc{ 1, 2, 3, 4 };                                              \
	float4 f4_desc{ 4, 3, 2, 1 };                                             \
	float8 f8_asc{ 1, 2, 3, 4, 5, 6, 7, 8 };                                  \
	float8 f8_desc{ 8, 7, 6, 5, 4, 3, 2, 1 };                                 \
	float16 f16_asc{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; \
	float16 f16_desc{ 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

#define VCL_SIMD_INTS                                                       \
	using int4 = Vcl::int4;                                                 \
	using int8 = Vcl::int8;                                                 \
	using int16 = Vcl::int16;                                               \
	int4 i4_asc{ 1, 2, 3, 4 };                                              \
	int4 i4_desc{ 4, 3, 2, 1 };                                             \
	int8 i8_asc{ 1, 2, 3, 4, 5, 6, 7, 8 };                                  \
	int8 i8_desc{ 8, 7, 6, 5, 4, 3, 2, 1 };                                 \
	int16 i16_asc{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; \
	int16 i16_desc{ 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

TEST(SimdBool, Construct)
{
	VCL_SIMD_BOOLS

	using Vcl::all;
	using Vcl::none;

	EXPECT_TRUE(all(bool4{ true }));
	EXPECT_TRUE(all(bool8{ true }));
	EXPECT_TRUE(all(bool16{ true }));

	EXPECT_TRUE(none(bool4{ false }));
	EXPECT_TRUE(none(bool8{ false }));
	EXPECT_TRUE(none(bool16{ false }));

	bool4 b4{ b4_0 };
	for (int i = 0; i < 4; i++)
	{
		EXPECT_EQ(b4_0[i], i % 2 == 0);
		EXPECT_EQ(b4[i], i % 2 == 0);
	}

	bool8 b8{ b8_0 };
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(b8_0[i], i % 2 == 0);
		EXPECT_EQ(b8[i], i % 2 == 0);
	}

	bool16 b16{ b16_0 };
	for (int i = 0; i < 16; i++)
	{
		EXPECT_EQ(b16_0[i], i % 2 == 0);
		EXPECT_EQ(b16[i], i % 2 == 0);
	}
}

TEST(SimdBool, Assign)
{
	VCL_SIMD_BOOLS

	using Vcl::all;

	bool4 b4{ true };
	bool8 b8{ true };
	bool16 b16{ true };

	EXPECT_TRUE(all(b4));
	EXPECT_TRUE(all(b8));
	EXPECT_TRUE(all(b16));

	b4 = b4_0;
	b8 = b8_0;
	b16 = b16_0;

	for (int i = 0; i < 4; i++)
		EXPECT_EQ(b4[i], i % 2 == 0);

	for (int i = 0; i < 8; i++)
		EXPECT_EQ(b8[i], i % 2 == 0);

	for (int i = 0; i < 16; i++)
		EXPECT_EQ(b16[i], i % 2 == 0);
}

TEST(SimdBool, And)
{
	VCL_SIMD_BOOLS

	using Vcl::none;

	bool4 b4 = b4_0 && b4_0;
	for (int i = 0; i < 4; i++)
		EXPECT_EQ(b4[i], i % 2 == 0);
	EXPECT_TRUE(none(b4_0 && b4_1));

	bool8 b8 = b8_0 && b8_0;
	for (int i = 0; i < 8; i++)
		EXPECT_EQ(b8[i], i % 2 == 0);
	EXPECT_TRUE(none(b8_0 && b8_1));

	bool16 b16 = b16_0 && b16_0;
	for (int i = 0; i < 16; i++)
		EXPECT_EQ(b16[i], i % 2 == 0);
	EXPECT_TRUE(none(b16_0 && b16_1));
}

TEST(SimdBool, Or)
{
	VCL_SIMD_BOOLS

	using Vcl::all;

	bool4 b4 = b4_0 || b4_0;
	for (int i = 0; i < 4; i++)
		EXPECT_EQ(b4[i], i % 2 == 0);
	EXPECT_TRUE(all(b4_0 || b4_1));

	bool8 b8 = b8_0 || b8_0;
	for (int i = 0; i < 8; i++)
		EXPECT_EQ(b8[i], i % 2 == 0);
	EXPECT_TRUE(all(b8_0 || b8_1));

	bool16 b16 = b16_0 || b16_0;
	for (int i = 0; i < 16; i++)
		EXPECT_EQ(b16[i], i % 2 == 0);
	EXPECT_TRUE(all(b16_0 || b16_1));
}

TEST(SimdFloat, Construct)
{
	VCL_SIMD_FLOATS

	float4 f4{ 1 };
	float4 f4_2{ f4 };
	for (int i = 0; i < 4; i++)
	{
		EXPECT_EQ(f4[i], 1);
		EXPECT_EQ(f4_2[i], 1);
		EXPECT_EQ(f4_asc[i], i + 1);
	}

	float8 f8{ 1 };
	float8 f8_2{ f8 };
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(f8[i], 1);
		EXPECT_EQ(f8_2[i], 1);
		EXPECT_EQ(f8_asc[i], i + 1);
	}

	float16 f16{ 1 };
	float16 f16_2{ f16 };
	for (int i = 0; i < 16; i++)
	{
		EXPECT_EQ(f16[i], 1);
		EXPECT_EQ(f16_2[i], 1);
		EXPECT_EQ(f16_asc[i], i + 1);
	}
}

TEST(SimdFloat, Assign)
{
	VCL_SIMD_FLOATS

	float4 f4{ 0 };
	for (int i = 0; i < 4; i++)
		EXPECT_EQ(f4[i], 0.0f);
	f4 = f4_asc;
	for (int i = 0; i < 4; i++)
		EXPECT_EQ(f4_asc[i], i + 1);

	float8 f8{ 0 };
	for (int i = 0; i < 8; i++)
		EXPECT_EQ(f8[i], 0.0f);
	f8 = f8_asc;
	for (int i = 0; i < 8; i++)
		EXPECT_EQ(f8_asc[i], i + 1);

	float16 f16{ 0 };
	for (int i = 0; i < 16; i++)
		EXPECT_EQ(f16[i], 0.0f);
	f16 = f16_asc;
	for (int i = 0; i < 16; i++)
		EXPECT_EQ(f16_asc[i], i + 1);
}

TEST(SimdFloat, Negate)
{
	VCL_SIMD_FLOATS

	float4 f4 = -f4_asc;
	for (int i = 0; i < 4; i++)
	{
		EXPECT_EQ(f4[i], -f4_asc[i]);
	}

	float8 f8 = -f8_asc;
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(f8[i], -f8_asc[i]);
	}

	float16 f16 = -f16_asc;
	for (int i = 0; i < 16; i++)
	{
		EXPECT_EQ(f16[i], -f16_asc[i]);
	}
}

TEST(SimdFloat, Add)
{
	VCL_SIMD_FLOATS

	float4 f4 = f4_asc + f4_asc;
	for (int i = 0; i < 4; i++)
	{
		EXPECT_EQ(f4[i], f4_asc[i] + f4_asc[i]);
	}

	float8 f8 = f8_asc + f8_asc;
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(f8[i], f8_asc[i] + f8_asc[i]);
	}

	float16 f16 = f16_asc + f16_asc;
	for (int i = 0; i < 16; i++)
	{
		EXPECT_EQ(f16[i], f16_asc[i] + f16_asc[i]);
	}
}

TEST(SimdFloat, Sub)
{
	VCL_SIMD_FLOATS

	float4 f4 = f4_asc - f4_asc;
	for (int i = 0; i < 4; i++)
	{
		EXPECT_EQ(f4[i], 0);
	}

	float8 f8 = f8_asc - f8_asc;
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(f8[i], 0);
	}

	float16 f16 = f16_asc - f16_asc;
	for (int i = 0; i < 16; i++)
	{
		EXPECT_EQ(f16[i], 0);
	}
}

TEST(SimdFloat, Inf)
{
	using Vcl::all;
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::isinf;

	const float inf = std::numeric_limits<float>::infinity();

	EXPECT_TRUE(all(isinf(float4(inf))));
	EXPECT_TRUE(all(float4(inf) == float4(inf)));
	EXPECT_TRUE(all(float4(-inf) == -float4(inf)));
	EXPECT_TRUE(all(float4(2 * inf) == float4(2) * float4(inf)));
	EXPECT_TRUE(none(isinf(float4(0) * float4(inf))));
}

TEST(SimdFloat, Sqrt)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::Mathematics::equal;

	// Source data
	float4 vec1{ 0.0f, 6.10116f, 11.6117f, 11.8436f };
	float8 vec2{ 0.0f, 6.10116f, 11.6117f, 11.8436f,
				 0.0f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 0.0f, 6.10116f, 11.6117f, 11.8436f,
				  0.0f, 6.10116f, 11.6117f, 11.8436f,
				  0.0f, 6.10116f, 11.6117f, 11.8436f,
				  0.0f, 6.10116f, 11.6117f, 11.8436f };

	// Compute 1 / sqrt(x)
	float4 res1 = sqrt(vec1);
	float8 res2 = sqrt(vec2);
	float16 res3 = sqrt(vec3);

	// Reference result
	float4 ref1{ 0.0f, 2.4700526f, 3.4075945f, 3.4414532f };
	float8 ref2{ 0.0f, 2.4700526f, 3.4075945f, 3.4414532f,
				 0.0f, 2.4700526f, 3.4075945f, 3.4414532f };
	float16 ref3{ 0.0f, 2.4700526f, 3.4075945f, 3.4414532f,
				  0.0f, 2.4700526f, 3.4075945f, 3.4414532f,
				  0.0f, 2.4700526f, 3.4075945f, 3.4414532f,
				  0.0f, 2.4700526f, 3.4075945f, 3.4414532f };

	EXPECT_TRUE(all(equal(ref1, res1, float4(1e-5f)))) << "'sqrt' failed.";
	EXPECT_TRUE(all(equal(ref2, res2, float8(1e-5f)))) << "'sqrt' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'sqrt' failed.";
}

template<int W>
void fltSignumTest()
{
	using Vcl::all;
	using Vcl::sgn;

	using floatN = Vcl::VectorScalar<float, W>;

	std::random_device rnd;
	std::uniform_real_distribution<float> dist{ -10, 10 };

	EXPECT_TRUE(all(sgn(floatN(0)) == floatN(0)));
	EXPECT_TRUE(all(sgn(floatN(-0.0f)) == floatN(0)));
	for (int i = 0; i < 50; i++)
	{
		const float d = dist(rnd);
		if (d < 0)
		{
			EXPECT_TRUE(all(sgn(floatN(d)) == floatN(-1)));
		} else if (d > 0)
		{
			EXPECT_TRUE(all(sgn(floatN(d)) == floatN(1)));
		}
	}
}

TEST(SimdFloat, Signum)
{
	fltSignumTest<4>();
	fltSignumTest<8>();
	fltSignumTest<16>();
}

template<typename T, int W>
void selectTest(
	const Vcl::VectorScalar<bool, W>& t,
	const Vcl::VectorScalar<T, W>& a,
	const Vcl::VectorScalar<T, W>& b,
	const Vcl::VectorScalar<T, W>& c)
{
	const auto selected = Vcl::select(t, a, b);
	EXPECT_TRUE(Vcl::all(selected == c));
}

TEST(SimdFloat, Select)
{
	VCL_SIMD_FLOATS

	selectTest<float, 4>(true, 1.0f, 0.0f, 1.0f);
	selectTest<float, 8>(true, 1.0f, 0.0f, 1.0f);
	selectTest<float, 16>(true, 1.0f, 0.0f, 1.0f);

	selectTest<float, 4>(false, 1.0f, 0.0f, 0.0f);
	selectTest<float, 8>(false, 1.0f, 0.0f, 0.0f);
	selectTest<float, 16>(false, 1.0f, 0.0f, 0.0f);

	selectTest<float, 4>(float4{ 1 } / float4{ 0 } < 0, 1.0f, 0.0f, 0.0f);
	selectTest<float, 8>(float8{ 1 } / float8{ 0 } < 0, 1.0f, 0.0f, 0.0f);
	selectTest<float, 16>(float16{ 1 } / float16{ 0 } < 0, 1.0f, 0.0f, 0.0f);
}

TEST(SimdFloat, Rsqrt)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::Mathematics::equal;

	// Source data
	float4 vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8 vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute 1 / sqrt(x)
	float4 res1 = rsqrt(vec1);
	float8 res2 = rsqrt(vec2);
	float16 res3 = rsqrt(vec3);

	// Reference result
	float4 ref1{ 0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f };
	float8 ref2{ 0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f,
				 0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f };
	float16 ref3{ 0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f,
				  0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f,
				  0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f,
				  0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f };

	EXPECT_TRUE(all(equal(ref1, res1, float4(1e-5f)))) << "'rsqrt' failed.";
	EXPECT_TRUE(all(equal(ref2, res2, float8(1e-5f)))) << "'rsqrt' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'rsqrt' failed.";
}

TEST(SimdFloat, Rcp)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::Mathematics::equal;

	// Source data
	float4 vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8 vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute 1 / x
	float4 res1 = rcp(vec1);
	float8 res2 = rcp(vec2);
	float16 res3 = rcp(vec3);

	// Reference result
	float4 ref1{ 0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f };
	float8 ref2{ 0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f,
				 0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f };
	float16 ref3{ 0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f,
				  0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f,
				  0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f,
				  0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f };

	EXPECT_TRUE(all(equal(ref1, res1, float4(1e-5f)))) << "'rcp' failed.";
	EXPECT_TRUE(all(equal(ref2, res2, float8(1e-5f)))) << "'rcp' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'rcp' failed.";
}

TEST(SimdFloat, Pow)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::Mathematics::equal;

	// Source data
	float4 vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8 vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute pow(x, 2)
	float4 res1 = pow(vec1, float4(2.0f));
	float8 res2 = pow(vec2, float8(2.0f));
	float16 res3 = pow(vec3, float16(2.0f));

	// Reference result
	float4 ref1{ 129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f };
	float8 ref2{ 129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f,
				 129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f };
	float16 ref3{ 129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f,
				  129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f,
				  129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f,
				  129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f };

	EXPECT_TRUE(all(equal(ref1, res1, float4(1e-5f)))) << "'pow' failed.";
	EXPECT_TRUE(all(equal(ref2, res2, float8(1e-5f)))) << "'pow' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'pow' failed.";
}

TEST(SimdFloat, Log)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::Mathematics::equal;

	// Source data
	float4 vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8 vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute log(x)
	float4 res1 = log(vec1);
	float8 res2 = log(vec2);
	float16 res3 = log(vec3);

	// Reference result
	float4 ref1{ 2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f };
	float8 ref2{ 2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f,
				 2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f };
	float16 ref3{ 2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f,
				  2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f,
				  2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f,
				  2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f };

	EXPECT_TRUE(all(equal(ref1, res1, float4(1e-5f)))) << "'log' failed.";
	EXPECT_TRUE(all(equal(ref2, res2, float8(1e-5f)))) << "'log' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'log' failed.";
}

TEST(SimdFloat, Abs)
{
	using Vcl::all;
	using Vcl::Mathematics::equal;

	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	// Source data
	float4 vec1{ -1 };
	float8 vec2{ -1 };
	float16 vec3{ -1 };

	// Compute dot(x, x)
	float4 res1 = abs(vec1);
	float8 res2 = abs(vec2);
	float16 res3 = abs(vec3);

	// Reference result
	EXPECT_TRUE(all(equal(res1, float4{ 1 })));
	EXPECT_TRUE(all(equal(res2, float8{ 1 })));
	EXPECT_TRUE(all(equal(res3, float16{ 1 })));
}

TEST(SimdFloat, Dot)
{
	using Vcl::Mathematics::equal;

	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	// Source data
	float4 vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8 vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f,
				  11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute dot(x, x)
	float res1 = vec1.dot(vec1);
	float res2 = vec2.dot(vec2);
	float res3 = vec3.dot(vec3);

	// Reference result
	float ref = 11.3805f * 11.3805f + 6.10116f * 6.10116f + 11.6117f * 11.6117f + 11.8436f * 11.8436f;

	EXPECT_TRUE(equal(ref * 1, res1, 1e-5f));
	EXPECT_TRUE(equal(ref * 2, res2, 1e-5f));
	EXPECT_TRUE(equal(ref * 4, res3, 1e-5f));
}

TEST(SimdFloat, HMinMax)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	float4 vec4(1, 2, 3, 4);
	EXPECT_EQ(vec4.min(), 1);
	EXPECT_EQ(vec4.max(), 4);

	float8 vec8(1, 2, 3, 4, 5, 6, 7, 8);
	EXPECT_EQ(vec8.min(), 1);
	EXPECT_EQ(vec8.max(), 8);

	float16 vec16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
	EXPECT_EQ(vec16.min(), 1);
	EXPECT_EQ(vec16.max(), 16);
}

TEST(SimdInt, Construct)
{
	VCL_SIMD_INTS

	int4 i4{ 1 };
	int4 i4_2{ i4 };
	for (int i = 0; i < 4; i++)
	{
		EXPECT_EQ(i4[i], 1);
		EXPECT_EQ(i4_2[i], 1);
		EXPECT_EQ(i4_asc[i], i + 1);
	}

	int8 i8{ 1 };
	int8 i8_2{ i8 };
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(i8[i], 1);
		EXPECT_EQ(i8_2[i], 1);
		EXPECT_EQ(i8_asc[i], i + 1);
	}

	int16 i16{ 1 };
	int16 i16_2{ i16 };
	for (int i = 0; i < 16; i++)
	{
		EXPECT_EQ(i16[i], 1);
		EXPECT_EQ(i16_2[i], 1);
		EXPECT_EQ(i16_asc[i], i + 1);
	}
}

TEST(SimdInt, Assign)
{
	VCL_SIMD_INTS

	int4 i4{ 0 };
	for (int i = 0; i < 4; i++)
		EXPECT_EQ(i4[i], 0.0f);
	i4 = i4_asc;
	for (int i = 0; i < 4; i++)
		EXPECT_EQ(i4_asc[i], i + 1);

	int8 i8{ 0 };
	for (int i = 0; i < 8; i++)
		EXPECT_EQ(i8[i], 0.0f);
	i8 = i8_asc;
	for (int i = 0; i < 8; i++)
		EXPECT_EQ(i8_asc[i], i + 1);

	int16 i16{ 0 };
	for (int i = 0; i < 16; i++)
		EXPECT_EQ(i16[i], 0.0f);
	i16 = i16_asc;
	for (int i = 0; i < 16; i++)
		EXPECT_EQ(i16_asc[i], i + 1);
}

TEST(SimdInt, Abs)
{
	using Vcl::all;
	using Vcl::Mathematics::equal;

	using Vcl::int16;
	using Vcl::int4;
	using Vcl::int8;

	// Source data
	int4 vec1{ -1 };
	int8 vec2{ -1 };
	int16 vec3{ -1 };

	// Compute dot(x, x)
	int4 res1 = abs(vec1);
	int8 res2 = abs(vec2);
	int16 res3 = abs(vec3);

	// Reference result
	EXPECT_TRUE(all(res1 == int4{ 1 }));
	EXPECT_TRUE(all(res2 == int8{ 1 }));
	EXPECT_TRUE(all(res3 == int16{ 1 }));
}

TEST(SimdInt, Max)
{
	VCL_SIMD_INTS

	int4 i4 = max(i4_asc, i4_desc);
	for (int i = 0; i < 4; i++)
		EXPECT_EQ(i4[i], std::max(i4_asc[i], i4_desc[i]));

	int8 i8 = max(i8_asc, i8_desc);
	for (int i = 0; i < 8; i++)
		EXPECT_EQ(i8[i], std::max(i8_asc[i], i8_desc[i]));

	int16 i16 = max(i16_asc, i16_desc);
	for (int i = 0; i < 16; i++)
		EXPECT_EQ(i16[i], std::max(i16_asc[i], i16_desc[i]));
}

TEST(SimdInt, Min)
{
	VCL_SIMD_INTS

	int4 i4 = min(i4_asc, i4_desc);
	for (int i = 0; i < 4; i++)
		EXPECT_EQ(i4[i], std::min(i4_asc[i], i4_desc[i]));

	int8 i8 = min(i8_asc, i8_desc);
	for (int i = 0; i < 8; i++)
		EXPECT_EQ(i8[i], std::min(i8_asc[i], i8_desc[i]));

	int16 i16 = min(i16_asc, i16_desc);
	for (int i = 0; i < 16; i++)
		EXPECT_EQ(i16[i], std::min(i16_asc[i], i16_desc[i]));
}

template<int W>
void intSignumTest()
{
	using Vcl::all;
	using Vcl::sgn;

	using intN = Vcl::VectorScalar<int, W>;

	std::random_device rnd;
	std::uniform_int_distribution<int> dist{ -10, 10 };

	EXPECT_TRUE(all(sgn(intN(0)) == intN(0)));
	for (int i = 0; i < 50; i++)
	{
		const int d = dist(rnd);
		if (d < 0)
		{
			EXPECT_TRUE(all(sgn(intN(d)) == intN(-1))) << d;
		} else if (d > 0)
		{
			EXPECT_TRUE(all(sgn(intN(d)) == intN(1))) << d;
		}
	}
}

TEST(SimdInt, Signum)
{
	intSignumTest<4>();
	intSignumTest<8>();
	intSignumTest<16>();
}

TEST(SimdInt, Select)
{
	selectTest<int, 4>(true, 1, 0, 1);
	selectTest<int, 8>(true, 1, 0, 1);
	selectTest<int, 16>(true, 1, 0, 1);

	selectTest<int, 4>(false, 1, 0, 0);
	selectTest<int, 8>(false, 1, 0, 0);
	selectTest<int, 16>(false, 1, 0, 0);
}
