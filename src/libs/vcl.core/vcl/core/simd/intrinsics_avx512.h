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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <cstdint>

#ifdef VCL_VECTORIZE_AVX512

#include <vcl/core/simd/intrinsics_avx.h>

#define VCL_M512I_SIGNBIT _mm512_set1_epi32(int(0x80000000))
#define VCL_M512I_ALLBITS _mm512_set1_epi32(int(0xffffffff))

namespace Vcl
{
	VCL_STRONG_INLINE __m512 _mm512_sgn_ps(__m512 v)
	{
		const __mmask16 is_eq_zero = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_EQ_OQ);
		return _mm512_castsi512_ps(_mm512_mask_set1_epi32
		(
			_mm512_or_epi32
			(
				_mm512_and_epi32(_mm512_castps_si512(v), VCL_M512I_SIGNBIT),
				_mm512_castps_si512(_mm512_set1_ps(1.0f))
			),
			is_eq_zero, 0
		));
	}

#if !defined(VCL_COMPILER_MSVC)
	__m512 _mm512_sin_ps(__m512 v);
	__m512 _mm512_cos_ps(__m512 v);
	__m512 _mm512_log_ps(__m512 v);
	__m512 _mm512_exp_ps(__m512 v);

	__m512 _mm512_acos_ps(__m512 v);
	__m512 _mm512_asin_ps(__m512 v);
	__m512 _mm512_atan2_ps(__m512 y, __m512 x);

	__m512 _mm512_pow_ps(__m512 x, __m512 y);
#endif

	VCL_STRONG_INLINE __mmask16 _mm512_cmpeq_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmpneq_ps(__m512 a, __m512 b) { return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmplt_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmple_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmpgt_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmpge_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ); }

	VCL_STRONG_INLINE __mmask16 _mm512_isinf_ps(__m512 x)
	{
		const __m512 sign_mask = _mm512_set1_ps(-0.0);
		const __m512 inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

		x = _mm512_andnot_ps(sign_mask, x);
		return _mm512_cmpeq_ps(x, inf);
	}

	VCL_STRONG_INLINE __m512 _mmVCL_rsqrt_ps(__m512 v)
	{
		const __m512 nr = _mm512_rsqrt14_ps(v);
		const __m512 muls = _mm512_mul_ps(_mm512_mul_ps(nr, nr), v);
		const __m512 beta = _mm512_mul_ps(_mm512_set1_ps(0.5f), nr);
		const __m512 gamma = _mm512_sub_ps(_mm512_set1_ps(3.0f), muls);

		return _mm512_mul_ps(beta, gamma);
	}

	VCL_STRONG_INLINE __m512 _mmVCL_rcp_ps(__m512 v)
	{
		const __m512 nr = _mm512_rcp14_ps(v);
		const __m512 muls = _mm512_mul_ps(_mm512_mul_ps(nr, nr), v);
		const __m512 dbl = _mm512_add_ps(nr, nr);

		// Filter out zero input to ensure
		const __mmask16 is_neq_zero_mask = _mm512_cmpneq_ps(v, _mm512_setzero_ps());
		const __m512 result = _mm512_mask_sub_ps(dbl, is_neq_zero_mask, dbl, muls);

		return result;
	}

	VCL_STRONG_INLINE float _mmVCL_hmin_ps(__m512 v)
	{
		return _mm512_reduce_min_ps(v);
	}

	VCL_STRONG_INLINE float _mmVCL_hmax_ps(__m512 v)
	{
		return _mm512_reduce_max_ps(v);
	}

	VCL_STRONG_INLINE float _mmVCL_dp_ps(__m512 a, __m512 b)
	{
		return _mm512_reduce_add_ps(_mm512_mul_ps(a, b));
	}

	VCL_STRONG_INLINE float _mmVCL_extract_ps(__m512 v, int i)
	{
		typedef union
		{
			__m512 x;
			float a[16];
		} F32;

		return F32{ v }.a[i];
	}

	VCL_STRONG_INLINE int _mmVCL_extract_epi32(__m512i v, int i)
	{
		typedef union
		{
			__m512i x;
			int32_t a[16];
		} U32;

		return U32{ v }.a[i];
	}
}
#endif // VCL_VECTORIZE_AVX512
