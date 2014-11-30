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
#include <vcl/core/simd/intrinsics_avx.h>

#ifdef VCL_VECTORIZE_AVX
VCL_BEGIN_EXTERNAL_HEADERS
#if defined VCL_VECTORIZE_AVX2 && !defined __AVX2__ 
#	define __AVX2__
#endif
#include <vcl/core/simd/avx_mathfun.h>
VCL_END_EXTERNAL_HEADERS

namespace Vcl
{
	__m256 _mm256_sin_ps(__m256 v)
	{
		return sin256_ps(v);
	}

	__m256 _mm256_cos_ps(__m256 v)
	{
		return cos256_ps(v);
	}

	__m256 _mm256_log_ps(__m256 v)
	{
		return log256_ps(v);
	}

	__m256 _mm256_exp_ps(__m256 v)
	{
		return exp256_ps(v);
	}

	__m256 _mm256_pow_ps(__m256 x, __m256 y)
	{
		return _mm256_exp_ps(_mm256_mul_ps(_mm256_log_ps(x), y));
	}
}
#endif /* VCL_VECTORIZE_AVX */
