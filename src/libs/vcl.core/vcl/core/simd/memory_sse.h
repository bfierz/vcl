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

// VCL
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl
{
#if defined VCL_VECTORIZE_SSE || defined VCL_VECTORIZE_AVX

	// https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
	VCL_STRONG_INLINE void load
	(
		__m128& x, __m128& y, __m128& z,
		const Eigen::Vector3f* base
	)
	{
		const float* p = base->data();
		__m128 x0y0z0x1 = _mm_load_ps(p + 0);
		__m128 y1z1x2y2 = _mm_load_ps(p + 4);
		__m128 z2x3y3z3 = _mm_load_ps(p + 8);
		__m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2, z2x3y3z3, _MM_SHUFFLE(2, 1, 3, 2));
		__m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1, y1z1x2y2, _MM_SHUFFLE(1, 0, 2, 1));
		x = _mm_shuffle_ps(x0y0z0x1, x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0)); // x0x1x2x3
		y = _mm_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2, 0)); // y0y1y2y3
		z = _mm_shuffle_ps(y0z0y1z1, z2x3y3z3, _MM_SHUFFLE(3, 0, 3, 1)); // z0z1z2z3
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const __m128& x, const __m128& y, const __m128& z
	)
	{
		__m128 x0x2y0y2 = _mm_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 y1y3z1z3 = _mm_shuffle_ps(y, z, _MM_SHUFFLE(3, 1, 3, 1));
		__m128 z0z2x1x3 = _mm_shuffle_ps(z, x, _MM_SHUFFLE(3, 1, 2, 0));

		__m128 rx0y0z0x1 = _mm_shuffle_ps(x0x2y0y2, z0z2x1x3, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 ry1z1x2y2 = _mm_shuffle_ps(y1y3z1z3, x0x2y0y2, _MM_SHUFFLE(3, 1, 2, 0));
		__m128 rz2x3y3z3 = _mm_shuffle_ps(z0z2x1x3, y1y3z1z3, _MM_SHUFFLE(3, 1, 3, 1));

		float* p = base->data();
		_mm_store_ps(p + 0, rx0y0z0x1);
		_mm_store_ps(p + 4, ry1z1x2y2);
		_mm_store_ps(p + 8, rz2x3y3z3);
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float4, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		__m128 x0, y0, z0;
		load(x0, y0, z0, base);

		loaded =
		{
			float4(x0),
			float4(y0),
			float4(z0)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int4, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		__m128 x0, y0, z0;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));

		loaded =
		{
			int4{ _mm_castps_si128(x0) },
			int4{ _mm_castps_si128(y0) },
			int4{ _mm_castps_si128(z0) }
		};
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float4, 3, 1>& value
	)
	{
		store(base, (__m128) value(0), (__m128) value(1), (__m128) value(2));
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int4, 3, 1>& value
	)
	{
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base),
			_mm_castsi128_ps((__m128i) value(0)),
			_mm_castsi128_ps((__m128i) value(1)),
			_mm_castsi128_ps((__m128i) value(2))
		);
	}
#endif // defined VCL_VECTORIZE_SSE || defined VCL_VECTORIZE_AVX

#if defined VCL_VECTORIZE_SSE && !defined VCL_VECTORIZE_AVX
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float8, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		__m128 x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, base);
		load(x1, y1, z1, base + 4);

		loaded =
		{
			float8(x0, x1),
			float8(y0, y1),
			float8(z0, z1)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int8, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		__m128 x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));
		load(x1, y1, z1, reinterpret_cast<const Eigen::Vector3f*>(base) +4);

		loaded =
		{
			int8{ _mm_castps_si128(x0), _mm_castps_si128(x1) },
			int8{ _mm_castps_si128(y0), _mm_castps_si128(y1) },
			int8{ _mm_castps_si128(z0), _mm_castps_si128(z1) }
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float16, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		__m128 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
		load(x0, y0, z0, base);
		load(x1, y1, z1, base + 4);
		load(x2, y2, z2, base + 8);
		load(x3, y3, z3, base + 12);

		loaded =
		{
			float16(x0, x1, x2, x3),
			float16(y0, y1, y2, y3),
			float16(z0, z1, z2, z3)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int16, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		__m128 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));
		load(x1, y1, z1, reinterpret_cast<const Eigen::Vector3f*>(base) +4);
		load(x2, y2, z2, reinterpret_cast<const Eigen::Vector3f*>(base) +8);
		load(x3, y3, z3, reinterpret_cast<const Eigen::Vector3f*>(base) +12);

		loaded =
		{
			int16{ _mm_castps_si128(x0), _mm_castps_si128(x1), _mm_castps_si128(x2), _mm_castps_si128(x3) },
			int16{ _mm_castps_si128(y0), _mm_castps_si128(y1), _mm_castps_si128(y2), _mm_castps_si128(y3) },
			int16{ _mm_castps_si128(z0), _mm_castps_si128(z1), _mm_castps_si128(z2), _mm_castps_si128(z3) }
		};
	}
#endif // defined VCL_VECTORIZE_SSE && !defined VCL_VECTORIZE_AVX
}
