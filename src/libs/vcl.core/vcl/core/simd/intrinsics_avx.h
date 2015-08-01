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

#ifdef VCL_VECTORIZE_AVX

#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	VCL_STRONG_INLINE __m256 _mm256_abs_ps(__m256 v)
	{
		return _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)), v);
	}
	VCL_STRONG_INLINE __m256 _mm256_sgn_ps(__m256 v)
	{
		return _mm256_and_ps(_mm256_or_ps(_mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000))), _mm256_set1_ps(1.0f)), _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_NEQ_OQ));
	}

#ifdef VCL_VECTORIZE_AVX2
	VCL_STRONG_INLINE __m256i _mm256_cmplt_epi32(__m256i a, __m256i b)
	{
		return _mm256_cmpgt_epi32(b, a);
	}
	VCL_STRONG_INLINE __m256i _mm256_cmpneq_epi32(__m256i a, __m256i b)
	{
		return _mm256_andnot_si256(_mm256_cmpeq_epi32(a, b), _mm256_set1_epi32(0xffffffff));
	}	
	VCL_STRONG_INLINE __m256i _mm256_cmple_epi32(__m256i a, __m256i b)
	{
		return _mm256_andnot_si256(_mm256_cmpgt_epi32(a, b), _mm256_set1_epi32(0xffffffff));
	}
	VCL_STRONG_INLINE __m256i _mm256_cmpge_epi32(__m256i a, __m256i b)
	{
		return _mm256_andnot_si256(_mm256_cmplt_epi32(a, b), _mm256_set1_epi32(0xffffffff));
	}
#endif // VCL_VECTORIZE_AVX2

	__m256 _mm256_sin_ps(__m256 v);	
	__m256 _mm256_cos_ps(__m256 v);
	__m256 _mm256_log_ps(__m256 v);
	__m256 _mm256_exp_ps(__m256 v);

	__m256 _mm256_acos_ps(__m256 v);
	__m256 _mm256_asin_ps(__m256 v);

	__m256 _mm256_atan2_ps(__m256 y, __m256 x);
	__m256 _mm256_pow_ps(__m256 x, __m256 y);

	VCL_STRONG_INLINE __m256 _mm256_cmpeq_ps(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
	VCL_STRONG_INLINE __m256 _mm256_cmpneq_ps(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ); }
	VCL_STRONG_INLINE __m256 _mm256_cmplt_ps(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
	VCL_STRONG_INLINE __m256 _mm256_cmple_ps(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }
	VCL_STRONG_INLINE __m256 _mm256_cmpgt_ps(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
	VCL_STRONG_INLINE __m256 _mm256_cmpge_ps(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }

	VCL_STRONG_INLINE __m256 _mmVCL_rsqrt_ps(__m256 v)
	{
		const __m256 nr = _mm256_rsqrt_ps(v);
		const __m256 muls = _mm256_mul_ps(_mm256_mul_ps(nr, nr), v);
		const __m256 beta = _mm256_mul_ps(_mm256_set1_ps(0.5f), nr);
		const __m256 gamma = _mm256_sub_ps(_mm256_set1_ps(3.0f), muls);

		return _mm256_mul_ps(beta, gamma);
	}

	VCL_STRONG_INLINE __m256 _mmVCL_rcp_ps(__m256 v)
	{
		__m256 nr = _mm256_rcp_ps(v);
		__m256 muls = _mm256_mul_ps(_mm256_mul_ps(nr, nr), v);
		__m256 dbl = _mm256_add_ps(nr, nr);

		// Filter out zero input to ensure 
		__m256 mask = _mm256_cmpeq_ps(v, _mm256_setzero_ps());
		__m256 filtered = _mm256_andnot_ps(mask, muls);
		__m256 result = _mm256_sub_ps(dbl, filtered);

		return result;
	}

	VCL_STRONG_INLINE __m256i _mmVCL_add_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_add_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_add_epi32(x0, y0);
		__m128i z1 = _mm_add_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_sub_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_sub_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_sub_epi32(x0, y0);
		__m128i z1 = _mm_sub_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_mullo_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_mullo_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_mullo_epi32(x0, y0);
		__m128i z1 = _mm_mullo_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}

	VCL_STRONG_INLINE __m256i _mmVCL_max_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_add_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_max_epi32(x0, y0);
		__m128i z1 = _mm_max_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}

	VCL_STRONG_INLINE __m256i _mmVCL_abs_epi32(__m256i x)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_abs_epi32(x);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm_abs_epi32(x0);
		__m128i y1 = _mm_abs_epi32(x1);

		return _mm256_set_m128i(y1, y0);
#endif
	}

	VCL_STRONG_INLINE __m256i _mmVCL_cmpeq_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_cmpeq_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_cmpeq_epi32(x0, y0);
		__m128i z1 = _mm_cmpeq_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_cmpneq_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_cmpneq_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_cmpneq_epi32(x0, y0);
		__m128i z1 = _mm_cmpneq_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_cmplt_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_cmplt_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_cmplt_epi32(x0, y0);
		__m128i z1 = _mm_cmplt_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_cmple_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_cmple_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_cmple_epi32(x0, y0);
		__m128i z1 = _mm_cmple_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_cmpgt_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_cmpgt_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_cmpgt_epi32(x0, y0);
		__m128i z1 = _mm_cmpgt_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_cmpge_epi32(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_cmpge_epi32(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_cmpge_epi32(x0, y0);
		__m128i z1 = _mm_cmpge_epi32(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}

	VCL_STRONG_INLINE __m256i _mmVCL_and_si256(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_and_si256(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_and_si128(x0, y0);
		__m128i z1 = _mm_and_si128(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_andnot_si256(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_andnot_si256(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_andnot_si128(x0, y0);
		__m128i z1 = _mm_andnot_si128(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_or_si256(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_or_si256(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_or_si128(x0, y0);
		__m128i z1 = _mm_or_si128(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}
	VCL_STRONG_INLINE __m256i _mmVCL_xor_si256(__m256i x, __m256i y)
	{
#ifdef VCL_VECTORIZE_AVX2
		return _mm256_xor_si256(x, y);
#else
		__m128i x0 = _mm256_extractf128_si256(x, 0);
		__m128i x1 = _mm256_extractf128_si256(x, 1);

		__m128i y0 = _mm256_extractf128_si256(y, 0);
		__m128i y1 = _mm256_extractf128_si256(y, 1);

		__m128i z0 = _mm_xor_si128(x0, y0);
		__m128i z1 = _mm_xor_si128(x1, y1);

		return _mm256_set_m128i(z1, z0);
#endif
	}

	VCL_STRONG_INLINE float _mmVCL_hmin_ps(__m256 v)
	{
		__m256 hilo = _mm256_permute2f128_ps(v, v, 0x81);
		__m256 redux = _mm256_min_ps(v, hilo);
		redux = _mm256_min_ps(redux, _mm256_shuffle_ps(redux, redux, 0x0e));
		redux = _mm256_min_ps(redux, _mm256_shuffle_ps(redux, redux, 0x01));

		return _mm_cvtss_f32(_mm256_castps256_ps128(redux));
	}

	VCL_STRONG_INLINE float _mmVCL_hmax_ps(__m256 v)
	{
		__m256 hilo = _mm256_permute2f128_ps(v, v, 0x81);
		__m256 redux = _mm256_max_ps(v, hilo);
		redux = _mm256_max_ps(redux, _mm256_shuffle_ps(redux, redux, 0x0e));
		redux = _mm256_max_ps(redux, _mm256_shuffle_ps(redux, redux, 0x01));
		
		return _mm_cvtss_f32(_mm256_castps256_ps128(redux));
	}

	VCL_STRONG_INLINE float _mmVCL_extract_ps(__m256 v, int i)
	{
#if 1
		typedef union
		{
			__m256 x;
			float a[8];
		} F32;

		return F32{ v }.a[i];
#else
		float dest;

		__m128 half;
		switch (i / 4)
		{
		case 0:
			half = _mm256_extractf128_ps(v, 0);
			break;
		case 1:
			half = _mm256_extractf128_ps(v, 1);
			break;
		}

		switch (i % 4)
		{
		case 0:
			*((int*) &(dest)) = _mm_extract_ps(half, 0);
			break;
		case 1:
			*((int*) &(dest)) = _mm_extract_ps(half, 1);
			break;
		case 2:
			*((int*) &(dest)) = _mm_extract_ps(half, 2);
			break;
		case 3:
			*((int*) &(dest)) = _mm_extract_ps(half, 3);
			break;
		}

		return dest;
#endif
	}

	VCL_STRONG_INLINE int _mmVCL_extract_epi32(__m256i v, int i)
	{
#if 1
		typedef union
		{
			__m256i x;
			int32_t a[8];
		} U32;

		return U32{ v }.a[i];
#else
		int dest;

		__m128i half;
		switch (i / 4)
		{
		case 0:
			half = _mm256_extractf128_si256(v, 0);
			break;
		case 1:
			half = _mm256_extractf128_si256(v, 1);
			break;
		}

		switch (i % 4)
		{
		case 0:
			dest = _mm_extract_epi32(half, 0);
			break;
		case 1:
			dest = _mm_extract_epi32(half, 1);
			break;
		case 2:
			dest = _mm_extract_epi32(half, 2);
			break;
		case 3:
			dest = _mm_extract_epi32(half, 3);
			break;
		}

		return dest;
#endif
	}
}
#endif // VCL_VECTORIZE_AVX
