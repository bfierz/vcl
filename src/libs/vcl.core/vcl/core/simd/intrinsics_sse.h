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
#include <limits>

#if defined(VCL_VECTORIZE_SSE)

#define VCL_M128I_SIGNBIT _mm_set1_epi32(int(0x80000000))
#define VCL_M128I_ALLBITS _mm_set1_epi32(int(0xffffffff))

namespace Vcl
{
	VCL_STRONG_INLINE __m128 _mm_abs_ps(__m128 v)
	{
		// Compute abs using logical operations
		return _mm_andnot_ps(_mm_castsi128_ps(VCL_M128I_SIGNBIT), v);
		
		// Compute abs using shift operations
		//return _mm_castsi128_ps(_mm_srli_epi32(_mm_slli_epi32(_mm_castps_si128(v), 1), 1));
	}

	VCL_STRONG_INLINE __m128 _mm_sgn_ps(__m128 v)
	{
		return _mm_and_ps(_mm_or_ps(_mm_and_ps(v, _mm_castsi128_ps(VCL_M128I_SIGNBIT)), _mm_set1_ps(1.0f)), _mm_cmpneq_ps(v, _mm_setzero_ps()));
	}

	VCL_STRONG_INLINE __m128i _mm_cmpneq_epi32(__m128i a, __m128i b)
	{
		return _mm_andnot_si128(_mm_cmpeq_epi32(a, b), VCL_M128I_ALLBITS);
	}	
	VCL_STRONG_INLINE __m128i _mm_cmple_epi32(__m128i a, __m128i b)
	{
		return _mm_andnot_si128(_mm_cmpgt_epi32(a, b), VCL_M128I_ALLBITS);
	}
	VCL_STRONG_INLINE __m128i _mm_cmpge_epi32(__m128i a, __m128i b)
	{
		return _mm_andnot_si128(_mm_cmplt_epi32(a, b), VCL_M128I_ALLBITS);
	}

	//! Bitwise (mask ? a : b)
	VCL_STRONG_INLINE __m128i _mm_logical_bitwise_select(__m128i a, __m128i b, __m128i mask)
	{
		a = _mm_and_si128(a, mask);    // clear a where mask = 0
		b = _mm_andnot_si128(mask, b); // clear b where mask = 1
		a = _mm_or_si128(a, b);        // a = a OR b                         
		return a;
	}

	VCL_STRONG_INLINE __m128 _mm_isinf_ps(__m128 x)
	{
		const __m128 sign_mask = _mm_set1_ps(-0.0);
		const __m128 inf = _mm_set1_ps(std::numeric_limits<float>::infinity());

		x = _mm_andnot_ps(sign_mask, x);
		x = _mm_cmpeq_ps(x, inf);
		return x;
	}

	__m128i _mmVCL_abs_epi32(__m128i a);
	__m128i _mmVCL_max_epi32(__m128i a, __m128i b);

#if !defined(VCL_COMPILER_MSVC) || _MSC_VER < 1920
	__m128 _mm_sin_ps(__m128 v);	
	__m128 _mm_cos_ps(__m128 v);
	__m128 _mm_log_ps(__m128 v);
	__m128 _mm_exp_ps(__m128 v);

	__m128 _mm_acos_ps(__m128 v);
	__m128 _mm_asin_ps(__m128 v);
	__m128 _mm_atan2_ps(__m128 in_y, __m128 in_x);

	__m128 _mm_pow_ps(__m128 x, __m128 y);
#endif

	__m128 _mmVCL_floor_ps(__m128 x);

	VCL_STRONG_INLINE __m128i _mmVCL_mullo_epi32(__m128i a, __m128i b)
	{
#ifdef VCL_VECTORIZE_SSE4_1
		return _mm_mullo_epi32(a, b);
#else
		const __m128i tmp1 = _mm_mul_epu32(a, b); /* mul 2,0*/
		const __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4)); /* mul 3,1 */
		return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))); /* shuffle results to [63..0] and pack */
#endif
	}

	// AP-803 Newton-Raphson Method with Streaming SIMD Extensions
	VCL_STRONG_INLINE __m128 _mmVCL_rsqrt_ps(__m128 v)
	{
		const __m128 nr = _mm_rsqrt_ps(v);
		const __m128 muls = _mm_mul_ps(_mm_mul_ps(nr, nr), v);
		const __m128 beta = _mm_mul_ps(_mm_set1_ps(0.5f), nr);
		const __m128 gamma = _mm_sub_ps(_mm_set1_ps(3.0f), muls);

		return _mm_mul_ps(beta, gamma);
	}

	// AP-803 Newton-Raphson Method with Streaming SIMD Extensions
	VCL_STRONG_INLINE __m128 _mmVCL_rcp_ps(__m128 v)
	{
		const __m128 nr = _mm_rcp_ps(v);
		const __m128 muls = _mm_mul_ps(_mm_mul_ps(nr, nr), v);
		const __m128 dbl = _mm_add_ps(nr, nr);

		// Filter out zero input to ensure 
		const __m128 mask = _mm_cmpeq_ps(v, _mm_setzero_ps());
		const __m128 filtered = _mm_andnot_ps(mask, muls);
		const __m128 result = _mm_sub_ps(dbl, filtered);

		return result;
	}

	VCL_STRONG_INLINE float _mmVCL_hmin_ps(__m128 v)
	{
		const __m128 data = v;             /* [0, 1, 2, 3] */
		const __m128 low = _mm_movehl_ps(data, data); /* [2, 3, 2, 3] */
		const __m128 low_accum = _mm_min_ps(low, data); /* [0|2, 1|3, 2|2, 3|3] */
		const __m128 elem1 = _mm_shuffle_ps(low_accum, low_accum, _MM_SHUFFLE(1, 1, 1, 1)); /* [1|3, 1|3, 1|3, 1|3] */
		const __m128 accum = _mm_min_ss(low_accum, elem1);
		return _mm_cvtss_f32(accum);
	}

	VCL_STRONG_INLINE float _mmVCL_hmax_ps(__m128 v)
	{
		const __m128 data = v;             /* [0, 1, 2, 3] */
		const __m128 high = _mm_movehl_ps(data, data); /* [2, 3, 2, 3] */
		const __m128 high_accum = _mm_max_ps(high, data); /* [0|2, 1|3, 2|2, 3|3] */
		const __m128 elem1 = _mm_shuffle_ps(high_accum, high_accum, _MM_SHUFFLE(1, 1, 1, 1)); /* [1|3, 1|3, 1|3, 1|3] */
		const __m128 accum = _mm_max_ss(high_accum, elem1);
		return _mm_cvtss_f32(accum);
	}

	VCL_STRONG_INLINE float _mmVCL_dp_ps(__m128 a, __m128 b)
	{
		typedef union
		{
			__m128 x;
			float a[4];
		} F32;

#ifdef VCL_VECTORIZE_SSE4_1
		return F32{ _mm_dp_ps(a, b, 0xff) }.a[0];
#elif defined VCL_VECTORIZE_SSE3
		const __m128 ab = _mm_mul_ps(a, b);
		const __m128 dp = _mm_hadd_ps(ab, ab);
		return F32{ dp }.a[0] + F32{ dp }.a[1];
#else
		const __m128 ab = _mm_mul_ps(a, b);
		return F32{ ab }.a[0] + F32{ ab }.a[1] + F32{ ab }.a[2] + F32{ ab }.a[3];
#endif
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

#ifdef VCL_VECTORIZE_SSE4_1
#	define _mmVCL_insert_ps _mm_insert_ps
#else
	VCL_STRONG_INLINE __m128 _mmVCL_insert_ps(__m128 a, __m128 b, const int sel)
	{
		typedef union
		{
			__m128 x;
			float a[4];
		} F32;

		float tmp;
		int count_d, zmask;

		F32 A,B;
		A.x = a;
		B.x = b;

		tmp     = B.a[(sel & 0xC0)>>6]; // 0xC0 = sel[7:6]
		count_d = (sel & 0x30)>>4;      // 0x30 = sel[5:4]
		zmask   = sel & 0x0F;           // 0x0F = sel[3:0]

		A.a[count_d] = tmp;

		A.a[0] = (zmask & 0x1) ? 0 : A.a[0];
		A.a[1] = (zmask & 0x2) ? 0 : A.a[1];
		A.a[2] = (zmask & 0x4) ? 0 : A.a[2];
		A.a[3] = (zmask & 0x8) ? 0 : A.a[3];
		return A.x;
	}
#endif

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
