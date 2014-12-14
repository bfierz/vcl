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

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/bool8_avx.h>
#include <vcl/core/simd/float8_avx.h>

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

	// Handbook of Mathematical Functions
	// M. Abramowitz and I.A. Stegun, Ed.
	__m256 _mm256_acos_ps(__m256 v)
	{
		using bool8  = VectorScalar<bool, 8>;
		using float8 = VectorScalar<float, 8>;

		float8 x{ v };

		// Absolute error <= 6.7e-5
		float8 negate = select(x < 0, float8{ 1 }, float8{ 0 });

		x = x.abs();

		float8 ret = -0.0187293f;
		ret = ret * x;
		ret = ret + 0.0742610f;
		ret = ret * x;
		ret = ret - 0.2121144f;
		ret = ret * x;
		ret = ret + 1.5707288f;
		ret = ret * (1.0f - x).sqrt();
		ret = ret - 2.0f * negate * ret;
		return static_cast<__m256>(negate * 3.14159265358979f + ret);
	}

	// Handbook of Mathematical Functions
	// M. Abramowitz and I.A. Stegun, Ed.
	__m256 _mm256_asin_ps(__m256 v)
	{
		using bool8 = VectorScalar<bool, 8>;
		using float8 = VectorScalar<float, 8>;

		float8 x{ v };

		float8 negate = select(x < 0, float8{ 1 }, float8{ 0 });

		x = abs(x);
		float8 ret = -0.0187293f;
		ret *= x;
		ret += 0.0742610f;
		ret *= x;
		ret -= 0.2121144f;
		ret *= x;
		ret += 1.5707288f;
		ret = 3.14159265358979f * 0.5f - sqrt(1.0f - x)*ret;
		return static_cast<__m256>(ret - 2.0f * negate * ret);
	}

	__m256 _mm256_atan2_ps(__m256 in_y, __m256 in_x)
	{
		using bool8 = VectorScalar<bool, 8>;
		using float8 = VectorScalar<float, 8>;

		float8 t0, t1, t3, t4;

		float8 x{ in_x };
		float8 y{ in_y };

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

		return static_cast<__m256>(t3);
	}

	__m256 _mm256_pow_ps(__m256 x, __m256 y)
	{
		return _mm256_exp_ps(_mm256_mul_ps(_mm256_log_ps(x), y));
	}
}
#endif // VCL_VECTORIZE_AVX
