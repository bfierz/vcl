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

#if defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)
VCL_BEGIN_EXTERNAL_HEADERS
#define USE_SSE2
#include <vcl/core/simd/sse_mathfun.h>
VCL_END_EXTERNAL_HEADERS

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
	
	// The following implementations are taken from:
	// http://dss.stephanierct.com/DevBlog/?p=8
	
#ifndef _mm_floor_ps
	__m128 _mm_floor_ps(__m128 x)
	{
		__m128i v0 = _mm_setzero_si128();
		__m128i v1 = _mm_cmpeq_epi32(v0,v0);
		__m128i ji = _mm_srli_epi32( v1, 25);
		__m128 j = *(__m128*)&_mm_slli_epi32( ji, 23); //create vector 1.0f
		__m128i i = _mm_cvttps_epi32(x);
		__m128 fi = _mm_cvtepi32_ps(i);
		__m128 igx = _mm_cmpgt_ps(fi, x);
		j = _mm_and_ps(igx, j);
		return _mm_sub_ps(fi, j);
	}
#endif
}
#endif /* defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX) */
