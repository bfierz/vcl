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
namespace Vcl
{
#ifdef VCL_VECTORIZE_AVX2
	VCL_STRONG_INLINE __m128 gather(float const* base, __m128i vindex)
	{
		return _mm_i32gather_ps(base, vindex, 4);
	}

	VCL_STRONG_INLINE VectorScalar<float, 4> gather(float const * base, const VectorScalar<int, 4>& vindex)
	{
		__m128i idx = vindex.get(0);
		__m128 val = gather(base, idx);
		return VectorScalar<float, 4>(val);
	}

	VCL_STRONG_INLINE __m256 gather(float const* base, __m256i vindex)
	{
		return _mm256_i32gather_ps(base, vindex, 4);
	}
#else
	VCL_STRONG_INLINE __m256 gather(float const* base, __m256i vindex)
	{
		typedef union
		{
			__m256i x;
			int32_t a[8];
		} U32;

		U32 idx{ vindex };

		__m256 res = _mm256_set_ps
		(
			*(base + idx.a[7]), *(base + idx.a[6]),
			*(base + idx.a[5]), *(base + idx.a[4]),
		
			*(base + idx.a[3]), *(base + idx.a[2]),
			*(base + idx.a[1]), *(base + idx.a[0])
		);
		
		return res;
	}
#endif

	VCL_STRONG_INLINE VectorScalar<float, 8> gather(float const * base, const VectorScalar<int, 8>& vindex)
	{
		__m256i idx = static_cast<__m256i>(vindex);
		return VectorScalar<float, 8>(gather(base, idx));
	}

	VCL_STRONG_INLINE VectorScalar<float, 16> gather(float const * base, const VectorScalar<int, 16>& vindex)
	{
		return VectorScalar<float, 16>
		(
			gather(base, vindex.get(0)),
			gather(base, vindex.get(1))
		);
	}

	VCL_STRONG_INLINE void load(float8& value, const float* base)
	{
		value = float8{ _mm256_loadu_ps(base) };
	}

	VCL_STRONG_INLINE void load(int8& value, const int* base)
	{
		value = int8{ _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base)) };
	}

	VCL_STRONG_INLINE void load(float16& value, const float* base)
	{
		value = float16
		{
			_mm256_loadu_ps(base + 0),
			_mm256_loadu_ps(base + 8)
		};
	}

	VCL_STRONG_INLINE void load(int16& value, const int* base)
	{
		value = int16
		{
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 0)),
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 8))
		};
	}

	// The load/store implementation for vectors are directly from or based on:
	// https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
	VCL_STRONG_INLINE void load
	(
		__m256& x, __m256& y, __m256& z,
		const Eigen::Vector3f* base
	)
	{
		const float* p = base->data();
		__m256 m03;
		__m256 m14;
		__m256 m25;
		m03 = _mm256_castps128_ps256(_mm_loadu_ps(p + 0)); // load lower halves
		m14 = _mm256_castps128_ps256(_mm_loadu_ps(p + 4));
		m25 = _mm256_castps128_ps256(_mm_loadu_ps(p + 8));
		m03 = _mm256_insertf128_ps(m03, _mm_loadu_ps(p + 12), 1);  // load upper halves
		m14 = _mm256_insertf128_ps(m14, _mm_loadu_ps(p + 16), 1);
		m25 = _mm256_insertf128_ps(m25, _mm_loadu_ps(p + 20), 1);

		__m256 xy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2, 1, 3, 2)); // upper x's and y's 
		__m256 yz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1, 0, 2, 1)); // lower y's and z's
		x = _mm256_shuffle_ps(m03, xy, _MM_SHUFFLE(2, 0, 3, 0));
		y = _mm256_shuffle_ps(yz, xy, _MM_SHUFFLE(3, 1, 2, 0));
		z = _mm256_shuffle_ps(yz, m25, _MM_SHUFFLE(3, 0, 3, 1));

	}

	VCL_STRONG_INLINE void load
	(
		__m256& x, __m256& y, __m256& z, __m256& w,
		const Eigen::Vector4f* base
	)
	{
		const float* p = base->data();
		__m256 m04;
		__m256 m15;
		__m256 m26;
		__m256 m37;

		// Load the lower halves
		m04 = _mm256_castps128_ps256(_mm_loadu_ps(p +  0));
		m15 = _mm256_castps128_ps256(_mm_loadu_ps(p +  4));
		m26 = _mm256_castps128_ps256(_mm_loadu_ps(p +  8));
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

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const __m256& x, const __m256& y, const __m256& z
	)
	{
		__m256 rxy = _mm256_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
		__m256 ryz = _mm256_shuffle_ps(y, z, _MM_SHUFFLE(3, 1, 3, 1));
		__m256 rzx = _mm256_shuffle_ps(z, x, _MM_SHUFFLE(3, 1, 2, 0));

		__m256 r03 = _mm256_shuffle_ps(rxy, rzx, _MM_SHUFFLE(2, 0, 2, 0));
		__m256 r14 = _mm256_shuffle_ps(ryz, rxy, _MM_SHUFFLE(3, 1, 2, 0));
		__m256 r25 = _mm256_shuffle_ps(rzx, ryz, _MM_SHUFFLE(3, 1, 3, 1));

		float* p = base->data();
		_mm_storeu_ps(p +  0, _mm256_castps256_ps128(r03));
		_mm_storeu_ps(p +  4, _mm256_castps256_ps128(r14));
		_mm_storeu_ps(p +  8, _mm256_castps256_ps128(r25));
		_mm_storeu_ps(p + 12, _mm256_extractf128_ps(r03, 1));
		_mm_storeu_ps(p + 16, _mm256_extractf128_ps(r14, 1));
		_mm_storeu_ps(p + 20, _mm256_extractf128_ps(r25, 1));
	}

	
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float8, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		__m256 x0, y0, z0;
		load(x0, y0, z0, base);

		loaded =
		{
			float8(x0),
			float8(y0),
			float8(z0)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int8, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		__m256 x0, y0, z0;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));

		loaded =
		{
			int8{ _mm256_castps_si256(x0) },
			int8{ _mm256_castps_si256(y0) },
			int8{ _mm256_castps_si256(z0) }
		};
	}
	
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float8, 4, 1>& loaded,
		const Eigen::Vector4f* base
	)
	{
		__m256 x0, y0, z0, w0;
		load(x0, y0, z0, w0, base);

		loaded =
		{
			float8(x0),
			float8(y0),
			float8(z0),
			float8(w0)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int8, 4, 1>& loaded,
		const Eigen::Vector4i* base
	)
	{
		__m256 x0, y0, z0, w0;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));

		loaded =
		{
			int8{ _mm256_castps_si256(x0) },
			int8{ _mm256_castps_si256(y0) },
			int8{ _mm256_castps_si256(z0) },
			int8{ _mm256_castps_si256(w0) }
		};
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float8, 3, 1>& value
	)
	{
		store
		(
			base,
			value(0).get(0),
			value(1).get(0),
			value(2).get(0)
		);
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int8, 3, 1>& value
	)
	{
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base),
			_mm256_castsi256_ps(static_cast<__m256i>(value(0))),
			_mm256_castsi256_ps(static_cast<__m256i>(value(1))),
			_mm256_castsi256_ps(static_cast<__m256i>(value(2)))
		);
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float16, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		__m256 x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, base);
		load(x1, y1, z1, base + 8);

		loaded =
		{
			float16(x0, x1),
			float16(y0, y1),
			float16(z0, z1)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int16, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		__m256 x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));
		load(x1, y1, z1, reinterpret_cast<const Eigen::Vector3f*>(base) + 8);

		loaded =
		{
			int16{ _mm256_castps_si256(x0), _mm256_castps_si256(x1) },
			int16{ _mm256_castps_si256(y0), _mm256_castps_si256(y1) },
			int16{ _mm256_castps_si256(z0), _mm256_castps_si256(z1) }
		};
	}
	
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float16, 4, 1>& loaded,
		const Eigen::Vector4f* base
	)
	{
		__m256 x0, x1, y0, w0, y1, z0, z1, w1;
		load(x0, y0, z0, w0, base);
		load(x1, y1, z1, w1, base + 8);

		loaded =
		{
			float16(x0, x1),
			float16(y0, y1),
			float16(z0, z1),
			float16(w0, w1)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int16, 4, 1>& loaded,
		const Eigen::Vector4i* base
	)
	{
		__m256 x0, x1, y0, w0, y1, z0, z1, w1;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));
		load(x1, y1, z1, w1, reinterpret_cast<const Eigen::Vector4f*>(base) + 8);

		loaded =
		{
			int16{ _mm256_castps_si256(x0), _mm256_castps_si256(x1) },
			int16{ _mm256_castps_si256(y0), _mm256_castps_si256(y1) },
			int16{ _mm256_castps_si256(z0), _mm256_castps_si256(z1) },
			int16{ _mm256_castps_si256(w0), _mm256_castps_si256(w1) }
		};
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float16, 3, 1>& value
	)
	{
		store(base    , value(0).get(0), value(1).get(0), value(2).get(0));
		store(base + 8, value(0).get(1), value(1).get(1), value(2).get(1));
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int16, 3, 1>& value
	)
	{
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base),
			_mm256_castsi256_ps(value(0).get(0)),
			_mm256_castsi256_ps(value(1).get(0)),
			_mm256_castsi256_ps(value(2).get(0))
		);
		
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base) + 8,
			_mm256_castsi256_ps(value(0).get(1)),
			_mm256_castsi256_ps(value(1).get(1)),
			_mm256_castsi256_ps(value(2).get(1))
		);
	}

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
}
#endif // VCL_VECTORIZE_AVX
