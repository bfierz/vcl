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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <cstdint>

#if defined(VCL_VECTORIZE_SSE)
namespace Vcl
{
	VCL_STRONG_INLINE __m128 _mm_abs_ps(__m128 v)
	{
		// Compute abs using logical operations
		return _mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32(0x80000000)), v);
		
		// Compute abs using shift operations
		//return _mm_castsi128_ps(_mm_srli_epi32(_mm_slli_epi32(_mm_castps_si128(v), 1), 1));
	}

	VCL_STRONG_INLINE __m128 _mm_sgn_ps(__m128 v)
	{
		return _mm_and_ps(_mm_or_ps(_mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x80000000))), _mm_set1_ps(1.0f)), _mm_cmpneq_ps(v, _mm_setzero_ps()));
	}

	VCL_STRONG_INLINE __m128i _mm_cmpneq_epi32(__m128i a, __m128i b)
	{
		return _mm_andnot_si128(_mm_cmpeq_epi32(a, b), _mm_set1_epi32(0xffffffff));
	}	
	VCL_STRONG_INLINE __m128i _mm_cmple_epi32(__m128i a, __m128i b)
	{
		return _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(0xffffffff));
	}
	VCL_STRONG_INLINE __m128i _mm_cmpge_epi32(__m128i a, __m128i b)
	{
		return _mm_andnot_si128(_mm_cmplt_epi32(a, b), _mm_set1_epi32(0xffffffff));
	}

	__m128 _mm_sin_ps(__m128 v);	
	__m128 _mm_cos_ps(__m128 v);
	__m128 _mm_log_ps(__m128 v);
	__m128 _mm_exp_ps(__m128 v);

	__m128 _mm_acos_ps(__m128 v);
	__m128 _mm_asin_ps(__m128 v);

	__m128 _mm_atan2_ps(__m128 y, __m128 x);
	__m128 _mm_pow_ps(__m128 x, __m128 y);

	__m128 _mmVCL_floor_ps(__m128 v);

	VCL_STRONG_INLINE __m128i _mmVCL_mullo_epi32(__m128i a, __m128i b)
	{
#ifdef VCL_VECTORIZE_SSE4_1
		return _mm_mullo_epi32(a, b);
#else
		__m128i tmp1 = _mm_mul_epu32(a, b); /* mul 2,0*/
		__m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4)); /* mul 3,1 */
		return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))); /* shuffle results to [63..0] and pack */
#endif
	}

	VCL_STRONG_INLINE __m128 _mmVCL_rsqrt_ps(__m128 v)
	{
		const __m128 nr = _mm_rsqrt_ps(v);
		const __m128 muls = _mm_mul_ps(_mm_mul_ps(nr, nr), v);
		const __m128 beta = _mm_mul_ps(_mm_set1_ps(0.5f), nr);
		const __m128 gamma = _mm_sub_ps(_mm_set1_ps(3.0f), muls);

		return _mm_mul_ps(beta, gamma);
	}

	VCL_STRONG_INLINE float _mmVCL_extract_ps(__m128 v, int i)
	{
#if 1
		typedef union
		{
			__m128 x;
			float a[4];
		} F32;

		return F32 {v}.a[i];
#else
#ifdef VCL_VECTORIZE_SSE4_1
		float dest;

		switch (i)
		{
		case 0:
			*((int*) &(dest)) = _mm_extract_ps(v, 0);
			break;
		case 1:
			*((int*) &(dest)) = _mm_extract_ps(v, 1);
			break;
		case 2:
			*((int*) &(dest)) = _mm_extract_ps(v, 2);
			break;
		case 3:
			*((int*) &(dest)) = _mm_extract_ps(v, 3);
			break;
		}

		return dest;
#else
		// Shuffle v so that the element that you want is moved to the least-
		// significant element of the vector (v[0])
		switch (i)
		{
		case 0:
			v = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
			break;
		case 1:
			v = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
			break;
		case 2:
			v = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
			break;
		case 3:
			v = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
			break;
		}

		// Move the value in V[0] to "ret"
		float ret;
		_mm_store_ss(&ret, v);

		return ret;
#endif
#endif
	}

	VCL_STRONG_INLINE int _mmVCL_extract_epi32(__m128i v, int i)
	{
#if 1
		typedef union
		{
			__m128i x;
			int32_t a[4];
		} U32;

		return U32 {v}.a[i];
#else
#ifdef VCL_VECTORIZE_SSE4_1
		int dest;

		switch (i)
		{
		case 0:
			dest = _mm_extract_epi32(v, 0);
			break;
		case 1:
			dest = _mm_extract_epi32(v, 1);
			break;
		case 2:
			dest = _mm_extract_epi32(v, 2);
			break;
		case 3:
			dest = _mm_extract_epi32(v, 3);
			break;
		}

		return dest;
#else
		// Shuffle v so that the element that you want is moved to the least-
		// significant element of the vector (v[0])
		switch (i)
		{
		case 0:
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 0, 0, 0));
			break;
		case 1:
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1));
			break;
		case 2:
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 2, 2, 2));
			break;
		case 3:
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 3, 3));
			break;
		}

		// Move the value in V[0] to "ret"
		int ret;
		_mm_store_ss((float*) &ret, _mm_castsi128_ps(v));

		return ret;
#endif
#endif
	}
}
#endif // defined(VCL_VECTORIZE_SSE)
