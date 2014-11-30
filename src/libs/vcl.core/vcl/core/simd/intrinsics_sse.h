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

#if defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)
namespace Vcl
{
	VCL_STRONG_INLINE __m128 _mm_abs_ps(__m128 v)
	{
		return _mm_castsi128_ps(_mm_srli_epi32(_mm_slli_epi32(_mm_castps_si128(v), 1), 1));
	}

	VCL_STRONG_INLINE __m128 _mm_sgn_ps(__m128 v)
	{
		return _mm_and_ps(_mm_or_ps(_mm_and_ps(v, _mm_set1_ps(-0.0f)), _mm_set1_ps(1.0f)), _mm_cmpneq_ps(v, _mm_setzero_ps()));
	}

	__m128 _mm_sin_ps(__m128 v);	
	__m128 _mm_cos_ps(__m128 v);
	__m128 _mm_log_ps(__m128 v);
	__m128 _mm_exp_ps(__m128 v);
	
	__m128 _mm_pow_ps(__m128 x, __m128 y);
	
#ifndef _mm_floor_ps
	__m128 _mm_floor_ps(__m128 v);
#endif
}
#endif // defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)
