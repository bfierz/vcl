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

#if defined(VCL_VECTORIZE_AVX)
namespace Vcl {
#	ifdef VCL_VECTORIZE_AVX512
	VCL_STRONG_INLINE __m512 gather(float const* base, __m512i vindex)
	{
		return _mm512_i32gather_ps(vindex, base, 4);
	}
#	endif

#	ifdef VCL_VECTORIZE_AVX2
	VCL_STRONG_INLINE __m128 gather(float const* base, __m128i vindex)
	{
		return _mm_i32gather_ps(base, vindex, 4);
	}

	VCL_STRONG_INLINE VectorScalar<float, 4> gather(float const* base, const VectorScalar<int, 4>& vindex)
	{
		__m128i idx = vindex.get(0);
		__m128 val = gather(base, idx);
		return VectorScalar<float, 4>(val);
	}

	VCL_STRONG_INLINE __m256 gather(float const* base, __m256i vindex)
	{
		return _mm256_i32gather_ps(base, vindex, 4);
	}
#	else
	VCL_STRONG_INLINE __m256 gather(float const* base, __m256i vindex)
	{
		typedef union
		{
			__m256i x;
			int32_t a[8];
		} U32;

		U32 idx{ vindex };

		__m256 res = _mm256_set_ps(
			*(base + idx.a[7]), *(base + idx.a[6]),
			*(base + idx.a[5]), *(base + idx.a[4]),

			*(base + idx.a[3]), *(base + idx.a[2]),
			*(base + idx.a[1]), *(base + idx.a[0]));

		return res;
	}
#	endif

	VCL_STRONG_INLINE VectorScalar<float, 8> gather(float const* base, const VectorScalar<int, 8>& vindex)
	{
		__m256i idx = vindex.get(0);
		return VectorScalar<float, 8>(gather(base, idx));
	}

#	ifdef VCL_VECTORIZE_AVX512
	VCL_STRONG_INLINE VectorScalar<float, 16> gather(float const* base, const VectorScalar<int, 16>& vindex)
	{
		return VectorScalar<float, 16>(gather(base, vindex.get(0)));
	}
#	else
	VCL_STRONG_INLINE VectorScalar<float, 16> gather(float const* base, const VectorScalar<int, 16>& vindex)
	{
		return VectorScalar<float, 16>(
			gather(base, vindex.get(0)),
			gather(base, vindex.get(1)));
	}
#	endif

	VCL_STRONG_INLINE void load(float8& value, const float* base)
	{
		value = float8{ _mm256_loadu_ps(base) };
	}

	VCL_STRONG_INLINE void load(int8& value, const int* base)
	{
		value = int8{ _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base)) };
	}

#	ifdef VCL_VECTORIZE_AVX512
	VCL_STRONG_INLINE void load(float16& value, const float* base)
	{
		value = float16{
			_mm512_loadu_ps(base + 0)
		};
	}

	VCL_STRONG_INLINE void load(int16& value, const int* base)
	{
		value = int16{
			_mm512_loadu_si512(reinterpret_cast<const __m256i*>(base + 0))
		};
	}
#	else
	VCL_STRONG_INLINE void load(float16& value, const float* base)
	{
		value = float16{
			_mm256_loadu_ps(base + 0),
			_mm256_loadu_ps(base + 8)
		};
	}

	VCL_STRONG_INLINE void load(int16& value, const int* base)
	{
		value = int16{
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 0)),
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 8))
		};
	}
#	endif

#	ifdef VCL_VECTORIZE_AVX512
	VCL_STRONG_INLINE void load(
		__m512& x,
		__m512& y,
		const Eigen::Vector2f* base)
	{
		const float* p = base->data();
		const __m512 m0123 = _mm512_loadu_ps(p + 0);  // x0y0x1y1     x2y2x3y3     x4y4x5y5     x6y6x7y7
		const __m512 m4567 = _mm512_loadu_ps(p + 16); // x8y8x9y9 x10y10x11y11 x12y12x13y13 x14y14x15y15

		const __m512 m0246 = _mm512_shuffle_f32x4(m0123, m4567, _MM_SHUFFLE(2, 0, 2, 0));
		const __m512 m1357 = _mm512_shuffle_f32x4(m0123, m4567, _MM_SHUFFLE(3, 1, 3, 1));

		x = _mm512_shuffle_ps(m0246, m1357, _MM_SHUFFLE(2, 0, 2, 0));
		y = _mm512_shuffle_ps(m0246, m1357, _MM_SHUFFLE(3, 1, 3, 1));
	}

	VCL_STRONG_INLINE void load(
		__m512& x,
		__m512& y,
		__m512& z,
		const Eigen::Vector3f* base)
	{
		const float* p = base->data();
		const __m512 m0123 = _mm512_loadu_ps(p + 0);  //     x0y0z0x1     y1z1x2y2     z2x3y3z3     x4y4z4x5
		const __m512 m4567 = _mm512_loadu_ps(p + 16); //     y5z5x6y6     z6x7y7z7     x8y8z8x9   y9z9x10y10
		const __m512 m89ab = _mm512_loadu_ps(p + 32); // z10x11y11z11 x12y12z12x13 y13z13x14y14 z14x15y15z15

		const __m512 m679a = _mm512_shuffle_f32x4(m4567, m89ab, _MM_SHUFFLE(2, 1, 3, 2));
		const __m512 m1245 = _mm512_shuffle_f32x4(m0123, m4567, _MM_SHUFFLE(1, 0, 2, 1));

		const __m512 m0369 = _mm512_shuffle_f32x4(m0123, m679a, _MM_SHUFFLE(2, 0, 3, 0));
		const __m512 m147a = _mm512_shuffle_f32x4(m1245, m679a, _MM_SHUFFLE(3, 1, 2, 0));
		const __m512 m258b = _mm512_shuffle_f32x4(m1245, m89ab, _MM_SHUFFLE(3, 0, 3, 1));

		// Same code as m128 case as _mm512_shuffle_ps works on 128-bit lanes
		const __m512 x0y0z0x1 = m0369;
		const __m512 y1z1x2y2 = m147a;
		const __m512 z2x3y3z3 = m258b;
		const __m512 x2y2x3y3 = _mm512_shuffle_ps(y1z1x2y2, z2x3y3z3, _MM_SHUFFLE(2, 1, 3, 2));
		const __m512 y0z0y1z1 = _mm512_shuffle_ps(x0y0z0x1, y1z1x2y2, _MM_SHUFFLE(1, 0, 2, 1));
		x = _mm512_shuffle_ps(x0y0z0x1, x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0)); // x0x1x2x3
		y = _mm512_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2, 0)); // y0y1y2y3
		z = _mm512_shuffle_ps(y0z0y1z1, z2x3y3z3, _MM_SHUFFLE(3, 0, 3, 1)); // z0z1z2z3
	}

	VCL_STRONG_INLINE void load(
		__m512& x,
		__m512& y,
		__m512& z,
		__m512& w,
		const Eigen::Vector4f* base)
	{
		const float* p = base->data();
		const __m512 m0123 = _mm512_loadu_ps(p + 0);
		const __m512 m4567 = _mm512_loadu_ps(p + 16);
		const __m512 m89ab = _mm512_loadu_ps(p + 32);
		const __m512 mcdef = _mm512_loadu_ps(p + 48);

		const __m512 m0145 = _mm512_shuffle_f32x4(m0123, m4567, _MM_SHUFFLE(1, 0, 1, 0));
		const __m512 m89cd = _mm512_shuffle_f32x4(m89ab, mcdef, _MM_SHUFFLE(1, 0, 1, 0));
		const __m512 m2367 = _mm512_shuffle_f32x4(m0123, m4567, _MM_SHUFFLE(3, 2, 3, 2));
		const __m512 mabef = _mm512_shuffle_f32x4(m89ab, mcdef, _MM_SHUFFLE(3, 2, 3, 2));

		const __m512 m048c = _mm512_shuffle_f32x4(m0145, m89cd, _MM_SHUFFLE(2, 0, 2, 0));
		const __m512 m159d = _mm512_shuffle_f32x4(m0145, m89cd, _MM_SHUFFLE(3, 1, 3, 1));
		const __m512 m26ae = _mm512_shuffle_f32x4(m2367, mabef, _MM_SHUFFLE(2, 0, 2, 0));
		const __m512 m37bf = _mm512_shuffle_f32x4(m2367, mabef, _MM_SHUFFLE(3, 1, 3, 1));

		// Same code as m128 case as _mm512_shuffle_ps works on 128-bit lanes
		const __m512 m0 = m048c;
		const __m512 m1 = m159d;
		const __m512 m2 = m26ae;
		const __m512 m3 = m37bf;

		const __m512 xy0 = _mm512_shuffle_ps(m0, m1, _MM_SHUFFLE(1, 0, 1, 0));
		const __m512 xy1 = _mm512_shuffle_ps(m2, m3, _MM_SHUFFLE(1, 0, 1, 0));
		const __m512 zw0 = _mm512_shuffle_ps(m0, m1, _MM_SHUFFLE(3, 2, 3, 2));
		const __m512 zw1 = _mm512_shuffle_ps(m2, m3, _MM_SHUFFLE(3, 2, 3, 2));

		x = _mm512_shuffle_ps(xy0, xy1, _MM_SHUFFLE(2, 0, 2, 0));
		y = _mm512_shuffle_ps(xy0, xy1, _MM_SHUFFLE(3, 1, 3, 1));
		z = _mm512_shuffle_ps(zw0, zw1, _MM_SHUFFLE(2, 0, 2, 0));
		w = _mm512_shuffle_ps(zw0, zw1, _MM_SHUFFLE(3, 1, 3, 1));
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3f* base,
		const __m512& x,
		const __m512& y,
		const __m512& z)
	{
		const __m512 x0x2y0y2 = _mm512_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
		const __m512 y1y3z1z3 = _mm512_shuffle_ps(y, z, _MM_SHUFFLE(3, 1, 3, 1));
		const __m512 z0z2x1x3 = _mm512_shuffle_ps(z, x, _MM_SHUFFLE(3, 1, 2, 0));

		const __m512 rx0y0z0x1 = _mm512_shuffle_ps(x0x2y0y2, z0z2x1x3, _MM_SHUFFLE(2, 0, 2, 0)); // m0369
		const __m512 ry1z1x2y2 = _mm512_shuffle_ps(y1y3z1z3, x0x2y0y2, _MM_SHUFFLE(3, 1, 2, 0)); // m147a
		const __m512 rz2x3y3z3 = _mm512_shuffle_ps(z0z2x1x3, y1y3z1z3, _MM_SHUFFLE(3, 1, 3, 1)); // m258b

		const __m512 m0617 = _mm512_shuffle_f32x4(rx0y0z0x1, ry1z1x2y2, _MM_SHUFFLE(2, 0, 2, 0));
		const __m512 m4a5b = _mm512_shuffle_f32x4(ry1z1x2y2, rz2x3y3z3, _MM_SHUFFLE(3, 1, 3, 1));
		const __m512 m2839 = _mm512_shuffle_f32x4(rz2x3y3z3, rx0y0z0x1, _MM_SHUFFLE(3, 1, 2, 0));

		const __m512 m0123 = _mm512_shuffle_f32x4(m0617, m2839, _MM_SHUFFLE(2, 0, 2, 0));
		const __m512 m4567 = _mm512_shuffle_f32x4(m4a5b, m0617, _MM_SHUFFLE(3, 1, 2, 0));
		const __m512 m89ab = _mm512_shuffle_f32x4(m2839, m4a5b, _MM_SHUFFLE(3, 1, 3, 1));

		float* p = base->data();
		_mm512_storeu_ps(p + 0, m0123);
		_mm512_storeu_ps(p + 16, m4567);
		_mm512_storeu_ps(p + 32, m89ab);
	}
#	endif

	VCL_STRONG_INLINE void load(
		__m256& x,
		__m256& y,
		const Eigen::Vector2f* base)
	{
		const float* p = base->data();

		__m256 m02;                                        // x0y0x1y1 x4y4x5y5
		__m256 m13;                                        // x2y2x3y3 6x6yx7y7
		m02 = _mm256_castps128_ps256(_mm_loadu_ps(p + 0)); // load lower halves
		m13 = _mm256_castps128_ps256(_mm_loadu_ps(p + 4));
		m02 = _mm256_insertf128_ps(m02, _mm_loadu_ps(p + 8), 1); // load upper halves
		m13 = _mm256_insertf128_ps(m13, _mm_loadu_ps(p + 12), 1);

		x = _mm256_shuffle_ps(m02, m13, _MM_SHUFFLE(2, 0, 2, 0));
		y = _mm256_shuffle_ps(m02, m13, _MM_SHUFFLE(3, 1, 3, 1));
	}

	// The load/store implementation for vectors are directly from or based on:
	// https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
	VCL_STRONG_INLINE void load(
		__m256& x,
		__m256& y,
		__m256& z,
		const Eigen::Vector3f* base)
	{
		const float* p = base->data();
		__m256 m03;
		__m256 m14;
		__m256 m25;
		m03 = _mm256_castps128_ps256(_mm_loadu_ps(p + 0)); // load lower halves
		m14 = _mm256_castps128_ps256(_mm_loadu_ps(p + 4));
		m25 = _mm256_castps128_ps256(_mm_loadu_ps(p + 8));
		m03 = _mm256_insertf128_ps(m03, _mm_loadu_ps(p + 12), 1); // load upper halves
		m14 = _mm256_insertf128_ps(m14, _mm_loadu_ps(p + 16), 1);
		m25 = _mm256_insertf128_ps(m25, _mm_loadu_ps(p + 20), 1);

		__m256 xy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2, 1, 3, 2)); // upper x's and y's
		__m256 yz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1, 0, 2, 1)); // lower y's and z's
		x = _mm256_shuffle_ps(m03, xy, _MM_SHUFFLE(2, 0, 3, 0));
		y = _mm256_shuffle_ps(yz, xy, _MM_SHUFFLE(3, 1, 2, 0));
		z = _mm256_shuffle_ps(yz, m25, _MM_SHUFFLE(3, 0, 3, 1));
	}

	VCL_STRONG_INLINE void load(
		__m256& x,
		__m256& y,
		__m256& z,
		__m256& w,
		const Eigen::Vector4f* base)
	{
		const float* p = base->data();
		__m256 m04;
		__m256 m15;
		__m256 m26;
		__m256 m37;

		// Load the lower halves
		m04 = _mm256_castps128_ps256(_mm_loadu_ps(p + 0));
		m15 = _mm256_castps128_ps256(_mm_loadu_ps(p + 4));
		m26 = _mm256_castps128_ps256(_mm_loadu_ps(p + 8));
		m37 = _mm256_castps128_ps256(_mm_loadu_ps(p + 12));

		// Load upper halves
		m04 = _mm256_insertf128_ps(m04, _mm_loadu_ps(p + 16), 1);
		m15 = _mm256_insertf128_ps(m15, _mm_loadu_ps(p + 20), 1);
		m26 = _mm256_insertf128_ps(m26, _mm_loadu_ps(p + 24), 1);
		m37 = _mm256_insertf128_ps(m37, _mm_loadu_ps(p + 28), 1);

		__m256 xy0 = _mm256_shuffle_ps(m04, m15, _MM_SHUFFLE(1, 0, 1, 0));
		__m256 xy1 = _mm256_shuffle_ps(m26, m37, _MM_SHUFFLE(1, 0, 1, 0));
		__m256 zw0 = _mm256_shuffle_ps(m04, m15, _MM_SHUFFLE(3, 2, 3, 2));
		__m256 zw1 = _mm256_shuffle_ps(m26, m37, _MM_SHUFFLE(3, 2, 3, 2));

		x = _mm256_shuffle_ps(xy0, xy1, _MM_SHUFFLE(2, 0, 2, 0));
		y = _mm256_shuffle_ps(xy0, xy1, _MM_SHUFFLE(3, 1, 3, 1));
		z = _mm256_shuffle_ps(zw0, zw1, _MM_SHUFFLE(2, 0, 2, 0));
		w = _mm256_shuffle_ps(zw0, zw1, _MM_SHUFFLE(3, 1, 3, 1));
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3f* base,
		const __m256& x,
		const __m256& y,
		const __m256& z)
	{
		__m256 rxy = _mm256_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
		__m256 ryz = _mm256_shuffle_ps(y, z, _MM_SHUFFLE(3, 1, 3, 1));
		__m256 rzx = _mm256_shuffle_ps(z, x, _MM_SHUFFLE(3, 1, 2, 0));

		__m256 r03 = _mm256_shuffle_ps(rxy, rzx, _MM_SHUFFLE(2, 0, 2, 0));
		__m256 r14 = _mm256_shuffle_ps(ryz, rxy, _MM_SHUFFLE(3, 1, 2, 0));
		__m256 r25 = _mm256_shuffle_ps(rzx, ryz, _MM_SHUFFLE(3, 1, 3, 1));

		float* p = base->data();
		_mm_storeu_ps(p + 0, _mm256_castps256_ps128(r03));
		_mm_storeu_ps(p + 4, _mm256_castps256_ps128(r14));
		_mm_storeu_ps(p + 8, _mm256_castps256_ps128(r25));
		_mm_storeu_ps(p + 12, _mm256_extractf128_ps(r03, 1));
		_mm_storeu_ps(p + 16, _mm256_extractf128_ps(r14, 1));
		_mm_storeu_ps(p + 20, _mm256_extractf128_ps(r25, 1));
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float8, 2, 1>& loaded,
		const Eigen::Vector2f* base)
	{
		__m256 x0, y0;
		load(x0, y0, base);

		loaded = {
			float8(x0),
			float8(y0)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int8, 2, 1>& loaded,
		const Eigen::Vector2i* base)
	{
		__m256 x0, y0;
		load(x0, y0, reinterpret_cast<const Eigen::Vector2f*>(base));

		loaded = {
			int8{ _mm256_castps_si256(x0) },
			int8{ _mm256_castps_si256(y0) }
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float8, 3, 1>& loaded,
		const Eigen::Vector3f* base)
	{
		__m256 x0, y0, z0;
		load(x0, y0, z0, base);

		loaded = {
			float8(x0),
			float8(y0),
			float8(z0)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int8, 3, 1>& loaded,
		const Eigen::Vector3i* base)
	{
		__m256 x0, y0, z0;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));

		loaded = {
			int8{ _mm256_castps_si256(x0) },
			int8{ _mm256_castps_si256(y0) },
			int8{ _mm256_castps_si256(z0) }
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float8, 4, 1>& loaded,
		const Eigen::Vector4f* base)
	{
		__m256 x0, y0, z0, w0;
		load(x0, y0, z0, w0, base);

		loaded = {
			float8(x0),
			float8(y0),
			float8(z0),
			float8(w0)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int8, 4, 1>& loaded,
		const Eigen::Vector4i* base)
	{
		__m256 x0, y0, z0, w0;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));

		loaded = {
			int8{ _mm256_castps_si256(x0) },
			int8{ _mm256_castps_si256(y0) },
			int8{ _mm256_castps_si256(z0) },
			int8{ _mm256_castps_si256(w0) }
		};
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float8, 3, 1>& value)
	{
		store(
			base,
			value(0).get(0),
			value(1).get(0),
			value(2).get(0));
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int8, 3, 1>& value)
	{
		store(
			reinterpret_cast<Eigen::Vector3f*>(base),
			_mm256_castsi256_ps(value(0).get(0)),
			_mm256_castsi256_ps(value(1).get(0)),
			_mm256_castsi256_ps(value(2).get(0)));
	}

#	ifdef VCL_VECTORIZE_AVX512
	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 2, 1>& loaded,
		const Eigen::Vector2f* base)
	{
		__m512 x0, y0;
		load(x0, y0, base);

		loaded = {
			float16(x0),
			float16(y0)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 3, 1>& loaded,
		const Eigen::Vector3f* base)
	{
		__m512 x0, y0, z0;
		load(x0, y0, z0, base);

		loaded = {
			float16(x0),
			float16(y0),
			float16(z0)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 4, 1>& loaded,
		const Eigen::Vector4f* base)
	{
		__m512 x0, y0, z0, w0;
		load(x0, y0, z0, w0, base);

		loaded = {
			float16(x0),
			float16(y0),
			float16(z0),
			float16(w0)
		};
	}
#	else
	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 2, 1>& loaded,
		const Eigen::Vector2f* base)
	{
		__m256 x0, x1, y0, y1;
		load(x0, y0, base);
		load(x1, y1, base + 8);

		loaded = {
			float16(x0, x1),
			float16(y0, y1)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int16, 2, 1>& loaded,
		const Eigen::Vector2i* base)
	{
		__m256 x0, x1, y0, y1;
		load(x0, y0, reinterpret_cast<const Eigen::Vector2f*>(base));
		load(x1, y1, reinterpret_cast<const Eigen::Vector2f*>(base) + 8);

		loaded = {
			int16{ _mm256_castps_si256(x0), _mm256_castps_si256(x1) },
			int16{ _mm256_castps_si256(y0), _mm256_castps_si256(y1) }
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 3, 1>& loaded,
		const Eigen::Vector3f* base)
	{
		__m256 x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, base);
		load(x1, y1, z1, base + 8);

		loaded = {
			float16(x0, x1),
			float16(y0, y1),
			float16(z0, z1)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int16, 3, 1>& loaded,
		const Eigen::Vector3i* base)
	{
		__m256 x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));
		load(x1, y1, z1, reinterpret_cast<const Eigen::Vector3f*>(base) + 8);

		loaded = {
			int16{ _mm256_castps_si256(x0), _mm256_castps_si256(x1) },
			int16{ _mm256_castps_si256(y0), _mm256_castps_si256(y1) },
			int16{ _mm256_castps_si256(z0), _mm256_castps_si256(z1) }
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 4, 1>& loaded,
		const Eigen::Vector4f* base)
	{
		__m256 x0, x1, y0, w0, y1, z0, z1, w1;
		load(x0, y0, z0, w0, base);
		load(x1, y1, z1, w1, base + 8);

		loaded = {
			float16(x0, x1),
			float16(y0, y1),
			float16(z0, z1),
			float16(w0, w1)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int16, 4, 1>& loaded,
		const Eigen::Vector4i* base)
	{
		__m256 x0, x1, y0, w0, y1, z0, z1, w1;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));
		load(x1, y1, z1, w1, reinterpret_cast<const Eigen::Vector4f*>(base) + 8);

		loaded = {
			int16{ _mm256_castps_si256(x0), _mm256_castps_si256(x1) },
			int16{ _mm256_castps_si256(y0), _mm256_castps_si256(y1) },
			int16{ _mm256_castps_si256(z0), _mm256_castps_si256(z1) },
			int16{ _mm256_castps_si256(w0), _mm256_castps_si256(w1) }
		};
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float16, 3, 1>& value)
	{
		store(base, value(0).get(0), value(1).get(0), value(2).get(0));
		store(base + 8, value(0).get(1), value(1).get(1), value(2).get(1));
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int16, 3, 1>& value)
	{
		store(
			reinterpret_cast<Eigen::Vector3f*>(base),
			_mm256_castsi256_ps(value(0).get(0)),
			_mm256_castsi256_ps(value(1).get(0)),
			_mm256_castsi256_ps(value(2).get(0)));

		store(
			reinterpret_cast<Eigen::Vector3f*>(base) + 8,
			_mm256_castsi256_ps(value(0).get(1)),
			_mm256_castsi256_ps(value(1).get(1)),
			_mm256_castsi256_ps(value(2).get(1)));
	}
#	endif

	/*
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float8, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		for (int i = 0; i < 8; i++)
		{
			loaded.x()[i] = base[i].x();
			loaded.y()[i] = base[i].y();
			loaded.z()[i] = base[i].z();
		}
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int8, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		for (int i = 0; i < 8; i++)
		{
			loaded.x()[i] = base[i].x();
			loaded.y()[i] = base[i].y();
			loaded.z()[i] = base[i].z();
		}
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float16, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		for (int i = 0; i < 16; i++)
		{
			loaded.x()[i] = base[i].x();
			loaded.y()[i] = base[i].y();
			loaded.z()[i] = base[i].z();
		}
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int16, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		for (int i = 0; i < 16; i++)
		{
			loaded.x()[i] = base[i].x();
			loaded.y()[i] = base[i].y();
			loaded.z()[i] = base[i].z();
		}
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float8, 3, 1>& value
	)
	{
		for (int i = 0; i < 8; i++)
		{
			base[i].x() = value.x()[i];
			base[i].y() = value.y()[i];
			base[i].z() = value.z()[i];
		}
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int8, 3, 1>& value
	)
	{
		for (int i = 0; i < 8; i++)
		{
			base[i].x() = value.x()[i];
			base[i].y() = value.y()[i];
			base[i].z() = value.z()[i];
		}
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float16, 3, 1>& value
	)
	{
		for (int i = 0; i < 16; i++)
		{
			base[i].x() = value.x()[i];
			base[i].y() = value.y()[i];
			base[i].z() = value.z()[i];
		}
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int16, 3, 1>& value
	)
	{
		for (int i = 0; i < 16; i++)
		{
			base[i].x() = value.x()[i];
			base[i].y() = value.y()[i];
			base[i].z() = value.z()[i];
		}
	}*/

	VCL_STRONG_INLINE std::array<float8, 2> interleave(const float8& a, const float8& b) noexcept
	{
		const __m256 low{ _mm256_unpacklo_ps(a.get(0), b.get(0)) };
		const __m256 high{ _mm256_unpackhi_ps(a.get(0), b.get(0)) };
		const __m256 l{ _mm256_permute2f128_ps(low, high, 0x20) };
		const __m256 h{ _mm256_permute2f128_ps(low, high, 0x31) };

		return { float8{ l }, float8{ h } };
	}
#	ifdef VCL_VECTORIZE_AVX512
	VCL_STRONG_INLINE std::array<float16, 2> interleave(const float16& a, const float16& b) noexcept
	{
		const int16 idx0{ 0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17 };
		const int16 idx1{ 0x08, 0x18, 0x09, 0x19, 0x0a, 0x1a, 0x0b, 0x1b, 0x0c, 0x1c, 0x0d, 0x1d, 0x0e, 0x1e, 0x0f, 0x1f };
		const __m512 low = _mm512_permutex2var_ps(a.get(0), idx0.get(0), b.get(0));
		const __m512 high = _mm512_permutex2var_ps(a.get(0), idx1.get(0), b.get(0));

		return { float16{ low }, float16{ high } };
	}
#	else
	VCL_STRONG_INLINE std::array<float16, 2> interleave(const float16& a, const float16& b) noexcept
	{
		const float16 low{ _mm256_unpacklo_ps(a.get(0), b.get(0)), _mm256_unpackhi_ps(a.get(0), b.get(0)) };
		const float16 high{ _mm256_unpacklo_ps(a.get(1), b.get(1)), _mm256_unpackhi_ps(a.get(1), b.get(1)) };
		const float16 l{ _mm256_permute2f128_ps(low.get(0), low.get(1), 0x20), _mm256_permute2f128_ps(low.get(0), low.get(1), 0x31) };
		const float16 h{ _mm256_permute2f128_ps(high.get(0), high.get(1), 0x20), _mm256_permute2f128_ps(high.get(0), high.get(1), 0x31) };

		return { l, h };
	}
#	endif
}
#endif // VCL_VECTORIZE_AVX
