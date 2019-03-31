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

TEST(Simd, Inf)
{
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;
	using Vcl::all;
	using Vcl::isinf;

	const float inf = std::numeric_limits<float>::infinity();

	EXPECT_TRUE(all(isinf(float4(inf))));
	EXPECT_TRUE(all(float4(inf) == float4(inf)));
	EXPECT_TRUE(all(float4(-inf) == -float4(inf)));
	EXPECT_TRUE(all(float4(2*inf) == float4(2)*float4(inf)));
	EXPECT_TRUE(none(isinf(float4(0)*float4(inf))));
}

TEST(Simd, Rsqrt)
{
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;
	
	using Vcl::Mathematics::equal;

	// Source data
	float4  vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8  vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute 1 / sqrt(x)
	float4  res1 = rsqrt(vec1);
	float8  res2 = rsqrt(vec2);
	float16 res3 = rsqrt(vec3);

	// Reference result
	float4  ref1{ 0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f };
	float8  ref2{ 0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f,
		          0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f };
	float16 ref3{ 0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f,
		          0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f,
		          0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f,
		          0.2964281f, 0.4048497f, 0.2934622f, 0.2905749f };

	EXPECT_TRUE(all(equal(ref1, res1,  float4(1e-5f)))) << "'rsqrt' failed.";
	EXPECT_TRUE(all(equal(ref2, res2,  float8(1e-5f)))) << "'rsqrt' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'rsqrt' failed.";
}

TEST(Simd, Rcp)
{
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;
	
	using Vcl::Mathematics::equal;

	// Source data
	float4  vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8  vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute 1 / x
	float4  res1 = rcp(vec1);
	float8  res2 = rcp(vec2);
	float16 res3 = rcp(vec3);

	// Reference result
	float4  ref1{ 0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f };
	float8  ref2{ 0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f,
		          0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f };
	float16 ref3{ 0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f,
		          0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f,
		          0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f,
		          0.0878696f, 0.16390326f, 0.0861200f, 0.0844338f };

	EXPECT_TRUE(all(equal(ref1, res1,  float4(1e-5f)))) << "'rcp' failed.";
	EXPECT_TRUE(all(equal(ref2, res2,  float8(1e-5f)))) << "'rcp' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'rcp' failed.";
}

TEST(Simd, Pow)
{
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;
	
	using Vcl::Mathematics::equal;

	// Source data
	float4  vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8  vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute pow(x, 2)
	float4  res1 = pow(vec1,  float4(2.0f));
	float8  res2 = pow(vec2,  float8(2.0f));
	float16 res3 = pow(vec3, float16(2.0f));

	// Reference result
	float4  ref1{ 129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f };
	float8  ref2{ 129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f,
		          129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f };
	float16 ref3{ 129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f,
		          129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f,
		          129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f,
		          129.51578025f, 37.2241533456f, 134.83157689f, 140.27086096f };

	EXPECT_TRUE(all(equal(ref1, res1,  float4(1e-5f)))) << "'pow' failed.";
	EXPECT_TRUE(all(equal(ref2, res2,  float8(1e-5f)))) << "'pow' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'pow' failed.";
}

TEST(Simd, Log)
{
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;
	
	using Vcl::Mathematics::equal;

	// Source data
	float4  vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8  vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float16 vec3{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f,
		          11.3805f, 6.10116f, 11.6117f, 11.8436f };

	// Compute log(x)
	float4  res1 = log(vec1);
	float8  res2 = log(vec2);
	float16 res3 = log(vec3);

	// Reference result
	float4  ref1{ 2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f };
	float8  ref2{ 2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f,
		          2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f };
	float16 ref3{ 2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f,
		          2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f,
		          2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f,
		          2.43190136f, 1.8084789170f, 2.45201321f, 2.47178763f };

	EXPECT_TRUE(all(equal(ref1, res1,  float4(1e-5f)))) << "'log' failed.";
	EXPECT_TRUE(all(equal(ref2, res2,  float8(1e-5f)))) << "'log' failed.";
	EXPECT_TRUE(all(equal(ref3, res3, float16(1e-5f)))) << "'log' failed.";
}

TEST(Simd, Dot)
{
	using Vcl::Mathematics::equal;

	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;

	// Source data
	float4  vec1{ 11.3805f, 6.10116f, 11.6117f, 11.8436f };
	float8  vec2{ 11.3805f, 6.10116f, 11.6117f, 11.8436f,
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
	float ref = 11.3805f*11.3805f + 6.10116f*6.10116f + 11.6117f*11.6117f + 11.8436f*11.8436f;

	EXPECT_TRUE(equal(ref*1, res1, 1e-5f));
	EXPECT_TRUE(equal(ref*2, res2, 1e-5f));
	EXPECT_TRUE(equal(ref*4, res3, 1e-5f));
}

TEST(Simd, HMinMax)
{
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;
	
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

TEST(Simd, Sign4)
{
	using Vcl::float4;
	using Vcl::all;
	using Vcl::sgn;

	std::random_device rnd;
	std::uniform_real_distribution<float> dist{ -10, 10 };

	EXPECT_TRUE(all(sgn(float4(0)) == float4(0)));
	EXPECT_TRUE(all(sgn(float4(-0.0)) == float4(0)));
	for (int i = 0; i < 50; i++)
	{
		const float d = dist(rnd);
		if (d < 0)
			EXPECT_TRUE(all(sgn(float4(d)) == float4(-1)));
		else
			EXPECT_TRUE(all(sgn(float4(d)) == float4(1)));
	}
}

TEST(Simd, Sign8)
{
	using Vcl::float8;
	using Vcl::all;
	using Vcl::sgn;

	std::random_device rnd;
	std::uniform_real_distribution<float> dist{ -10, 10 };

	EXPECT_TRUE(all(sgn(float8(0)) == float8(0)));
	EXPECT_TRUE(all(sgn(float8(-0.0)) == float8(0)));
	for (int i = 0; i < 50; i++)
	{
		const float d = dist(rnd);
		if (d < 0)
			EXPECT_TRUE(all(sgn(float8(d)) == float8(-1))) << d;
		else
			EXPECT_TRUE(all(sgn(float8(d)) == float8(1))) << d;
	}
}
