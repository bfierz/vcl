/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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
#include <vcl/core/simd/intrinsics_sse.h>

#if defined(VCL_VECTORIZE_SSE)
VCL_BEGIN_EXTERNAL_HEADERS
#define USE_SSE2
#include <vcl/core/simd/sse_mathfun.h>
VCL_END_EXTERNAL_HEADERS

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/bool4_sse.h>
#include <vcl/core/simd/float4_sse.h>

namespace Vcl
{

	__m128 _mm_sin_ps(__m128 v)
	{
		return sin_ps(v);
	}

	__m128 _mm_cos_ps(__m128 v)
	{
		return cos_ps(v);
	}

	__m128 _mm_log_ps(__m128 v)
	{
		return log_ps(v);
	}

	__m128 _mm_exp_ps(__m128 v)
	{
		return exp_ps(v);
	}
	
	__m128 _mm_pow_ps(__m128 x, __m128 y)
	{
		return _mm_exp_ps(_mm_mul_ps(_mm_log_ps(x), y));
	}

	// Handbook of Mathematical Functions
	// M. Abramowitz and I.A. Stegun, Ed.
	__m128 _mm_acos_ps(__m128 v)
	{
		using bool4  = VectorScalar<bool, 4>;
		using float4 = VectorScalar<float, 4>;

		float4 x{ v };

		// Absolute error <= 6.7e-5
		float4 negate = select(x < 0, float4{ 1 }, float4{ 0 });

		x = x.abs();

		float4 ret = -0.0187293f;
		ret = ret * x;
		ret = ret + 0.0742610f;
		ret = ret * x;
		ret = ret - 0.2121144f;
		ret = ret * x;
		ret = ret + 1.5707288f;
		ret = ret * (1.0f - x).sqrt();
		ret = ret - 2.0f * negate * ret;
		return static_cast<__m128>(negate * 3.14159265358979f + ret);
	}

	// Handbook of Mathematical Functions
	// M. Abramowitz and I.A. Stegun, Ed.
	__m128 _mm_asin_ps(__m128 v)
	{
		using bool4 = VectorScalar<bool, 4>;
		using float4 = VectorScalar<float, 4>;

		float4 x{ v };

		float4 negate = select(x < 0, float4{ 1 }, float4{ 0 });

		x = abs(x);
		float4 ret = -0.0187293f;
		ret *= x;
		ret += 0.0742610f;
		ret *= x;
		ret -= 0.2121144f;
		ret *= x;
		ret += 1.5707288f;
		ret = 3.14159265358979f * 0.5f - sqrt(1.0f - x)*ret;
		return static_cast<__m128>(ret - 2.0f * negate * ret);
	}


	__m128 _mm256_atan2_ps(__m128 in_y, __m128 in_x)
	{
		using bool4 = VectorScalar<bool, 4>;
		using float4 = VectorScalar<float, 4>;

		float4 t0, t1, t3, t4;

		float4 x{ in_x };
		float4 y{ in_y };

		t3 = abs(x);
		t1 = abs(y);
		t0 = max(t3, t1);
		t1 = min(t3, t1);
		t3 = 1.0f / t0;
		t3 = t1 * t3;

		t4 = t3 * t3;
		t0 = -0.013480470f;
		t0 = t0 * t4 + 0.057477314f;
		t0 = t0 * t4 - 0.121239071f;
		t0 = t0 * t4 + 0.195635925f;
		t0 = t0 * t4 - 0.332994597f;
		t0 = t0 * t4 + 0.999995630f;
		t3 = t0 * t3;

		t3 = select(abs(y) > abs(x), 1.570796327f - t3, t3);
		t3 = select(x < 0, 3.141592654f - t3, t3);
		t3 = select(y < 0, -t3, t3);

		return static_cast<__m128>(t3);
	}
	
	__m128 _mmVCL_floor_ps(__m128 x)
	{
#ifdef VCL_VECTORIZE_SSE4_1
		return _mm_floor_ps(x);
#else
		// The following implementations are taken from:
		// http://dss.stephanierct.com/DevBlog/?p=8

		__m128i v0 = _mm_setzero_si128();
		__m128i v1 = _mm_cmpeq_epi32(v0,v0);
		__m128i ji = _mm_srli_epi32( v1, 25);
		__m128 j = *(__m128*)&_mm_slli_epi32( ji, 23); //create vector 1.0f
		__m128i i = _mm_cvttps_epi32(x);
		__m128 fi = _mm_cvtepi32_ps(i);
		__m128 igx = _mm_cmpgt_ps(fi, x);
		j = _mm_and_ps(igx, j);
		return _mm_sub_ps(fi, j);
#endif
	}
}
#endif // defined(VCL_VECTORIZE_SSE)
