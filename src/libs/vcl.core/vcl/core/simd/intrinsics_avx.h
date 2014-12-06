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

#ifdef VCL_VECTORIZE_AVX
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

#if VCL_VECTORIZE_AVX_LEVEL_MAJOR >= 2
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
#endif // VCL_VECTORIZE_AVX_LEVEL_MAJOR >= 2

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
}
#endif // VCL_VECTORIZE_AVX
