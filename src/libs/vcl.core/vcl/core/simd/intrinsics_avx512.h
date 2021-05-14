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

namespace Vcl
{
	VCL_STRONG_INLINE __mmask16 _mm512_cmpeq_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmpneq_ps(__m512 a, __m512 b) { return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmplt_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmple_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmpgt_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
	VCL_STRONG_INLINE __mmask16 _mm512_cmpge_ps(__m512 a, __m512 b)  { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ); }

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
