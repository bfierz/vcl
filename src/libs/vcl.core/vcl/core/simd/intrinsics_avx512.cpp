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
#include <vcl/core/simd/intrinsics_avx512.h>

#if defined VCL_VECTORIZE_AVX512
VCL_BEGIN_EXTERNAL_HEADERS
#if defined VCL_VECTORIZE_AVX2 && !defined __AVX2__
#	define __AVX2__
#endif
#include <vcl/core/simd/detail/avx512_mathfun.h>
VCL_END_EXTERNAL_HEADERS

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/bool16_avx512.h>
#include <vcl/core/simd/float16_avx512.h>

namespace Vcl
{
#if !defined(VCL_COMPILER_MSVC)
	__m512 _mm512_sin_ps(__m512 v)
	{
		return sin512_ps(v);
	}

	__m512 _mm512_cos_ps(__m512 v)
	{
		return cos512_ps(v);
	}

	__m512 _mm512_log_ps(__m512 v)
	{
		return log512_ps(v);
	}

	__m512 _mm512_exp_ps(__m512 v)
	{
		return exp512_ps(v);
	}

	// Handbook of Mathematical Functions
	// M. Abramowitz and I.A. Stegun, Ed.
	__m512 _mm512_acos_ps(__m512 v)
	{
		float16 x{ v };

		// Absolute error <= 6.7e-5
		float16 negate = select(x < 0, float16{ 1 }, float16{ 0 });

		x = x.abs();

		float16 ret = -0.0187293f;
		ret = ret * x;
		ret = ret + 0.0742610f;
		ret = ret * x;
		ret = ret - 0.2121144f;
		ret = ret * x;
		ret = ret + 1.5707288f;
		ret = ret * (1.0f - x).sqrt();
		ret = ret - 2.0f * negate * ret;
		return (negate * 3.14159265358979f + ret).get(0);
	}

	// Handbook of Mathematical Functions
	// M. Abramowitz and I.A. Stegun, Ed.
	__m512 _mm512_asin_ps(__m512 v)
	{
		float16 x{ v };

		float16 negate = select(x < 0, float16{ 1 }, float16{ 0 });

		x = abs(x);
		float16 ret = -0.0187293f;
		ret *= x;
		ret += 0.0742610f;
		ret *= x;
		ret -= 0.2121144f;
		ret *= x;
		ret += 1.5707288f;
		ret = 3.14159265358979f * 0.5f - sqrt(1.0f - x)*ret;
		return (ret - 2.0f * negate * ret).get(0);
	}

	__m512 _mm512_atan2_ps(__m512 in_y, __m512 in_x)
	{
		float16 t0, t1, t3, t4;

		float16 x{ in_x };
		float16 y{ in_y };

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

		return t3.get(0);
	}

	__m512 _mm512_pow_ps(__m512 x, __m512 y)
	{
		return _mm512_exp_ps(_mm512_mul_ps(_mm512_log_ps(x), y));
	}
#endif
}
#endif // VCL_VECTORIZE_AVX
