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

// C++ Standard Library
#include <array>

// VCL 
#include <vcl/core/simd/bool16_sse.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<int, 16>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() {}
		VCL_STRONG_INLINE VectorScalar(int s)
		{
			mF4[0] = _mm_set1_epi32(s);
			mF4[1] = _mm_set1_epi32(s);
			mF4[2] = _mm_set1_epi32(s);
			mF4[3] = _mm_set1_epi32(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar
		(
			int s00, int s01, int s02, int s03, int s04, int s05, int s06, int s07,
			int s08, int s09, int s10, int s11, int s12, int s13, int s14, int s15
		)
		{
			mF4[0] = _mm_set_epi32(s03, s02, s01, s00);
			mF4[1] = _mm_set_epi32(s07, s06, s05, s04);
			mF4[2] = _mm_set_epi32(s11, s10, s09, s08);
			mF4[3] = _mm_set_epi32(s15, s14, s13, s12);
		}
		VCL_STRONG_INLINE explicit VectorScalar(__m128i I4_0, __m128i I4_1, __m128i I4_2, const __m128i& I4_3)
		{
			mF4[0] = I4_0;
			mF4[1] = I4_1;
			mF4[2] = I4_2;
			mF4[3] = I4_3;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16>& operator = (const VectorScalar<int, 16>& rhs)
		{
			mF4[0] = rhs.mF4[0];
			mF4[1] = rhs.mF4[1];
			mF4[2] = rhs.mF4[2];
			mF4[3] = rhs.mF4[3];

			return *this;
		}

	public:
		VCL_STRONG_INLINE int operator[] (int idx) const
		{
			Require(0 <= idx && idx < 16, "Access is in range.");

			return _mmVCL_extract_epi32(mF4[idx / 4], idx % 4);
		}

		VCL_STRONG_INLINE __m128i get(int i) const
		{
			Require(0 <= i && i < 4, "Access is in range.");

			return mF4[i];
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> operator+ (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mm_add_epi32(mF4[0], rhs.mF4[0]),
				_mm_add_epi32(mF4[1], rhs.mF4[1]),
				_mm_add_epi32(mF4[2], rhs.mF4[2]),
				_mm_add_epi32(mF4[3], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator- (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mm_sub_epi32(mF4[0], rhs.mF4[0]),
				_mm_sub_epi32(mF4[1], rhs.mF4[1]),
				_mm_sub_epi32(mF4[2], rhs.mF4[2]),
				_mm_sub_epi32(mF4[3], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator* (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_mullo_epi32(mF4[0], rhs.mF4[0]),
				_mmVCL_mullo_epi32(mF4[1], rhs.mF4[1]),
				_mmVCL_mullo_epi32(mF4[2], rhs.mF4[2]),
				_mmVCL_mullo_epi32(mF4[3], rhs.mF4[3])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> abs()  const
		{
			return VectorScalar<int, 16>
			(
				_mm_abs_epi32(mF4[0]),
				_mm_abs_epi32(mF4[1]),
				_mm_abs_epi32(mF4[2]),
				_mm_abs_epi32(mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> max(const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mm_max_epi32(mF4[0], rhs.mF4[0]),
				_mm_max_epi32(mF4[1], rhs.mF4[1]),
				_mm_max_epi32(mF4[2], rhs.mF4[2]),
				_mm_max_epi32(mF4[3], rhs.mF4[3])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator== (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmpeq_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmpeq_epi32(mF4[1], rhs.mF4[1]),
				_mm_cmpeq_epi32(mF4[0], rhs.mF4[2]),
				_mm_cmpeq_epi32(mF4[1], rhs.mF4[3])
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 16> operator< (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmplt_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmplt_epi32(mF4[1], rhs.mF4[1]),
				_mm_cmplt_epi32(mF4[0], rhs.mF4[2]),
				_mm_cmplt_epi32(mF4[1], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator<= (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmple_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmple_epi32(mF4[1], rhs.mF4[1]),
				_mm_cmple_epi32(mF4[0], rhs.mF4[2]),
				_mm_cmple_epi32(mF4[1], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator> (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmpgt_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmpgt_epi32(mF4[1], rhs.mF4[1]),
				_mm_cmpgt_epi32(mF4[0], rhs.mF4[2]),
				_mm_cmpgt_epi32(mF4[1], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator>= (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmpge_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmpge_epi32(mF4[1], rhs.mF4[1]),
				_mm_cmpge_epi32(mF4[0], rhs.mF4[2]),
				_mm_cmpge_epi32(mF4[1], rhs.mF4[3])
			);
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 16>& rhs);
		friend VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b);
		friend VectorScalar<int, 16> signum(const VectorScalar<int, 16>& a);

	private:
		__m128i mF4[4];
	};
	
	VCL_STRONG_INLINE VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 16>
		(
			_mm_xor_si128(b.mF4[0], _mm_and_si128(_mm_castps_si128(mask.mF4[0]), _mm_xor_si128(b.mF4[0], a.mF4[0]))),
			_mm_xor_si128(b.mF4[1], _mm_and_si128(_mm_castps_si128(mask.mF4[1]), _mm_xor_si128(b.mF4[1], a.mF4[1]))),
			_mm_xor_si128(b.mF4[2], _mm_and_si128(_mm_castps_si128(mask.mF4[2]), _mm_xor_si128(b.mF4[2], a.mF4[2]))),
			_mm_xor_si128(b.mF4[3], _mm_and_si128(_mm_castps_si128(mask.mF4[3]), _mm_xor_si128(b.mF4[3], a.mF4[3])))
		);
	}

	VCL_STRONG_INLINE VectorScalar<int, 16> signum(const VectorScalar<int, 16>& a)
	{
		return VectorScalar<int, 16>
		(
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4[0], _mm_set1_epi32(0x80000000)), _mm_set1_epi32(1)
				), _mm_cmpneq_epi32(a.mF4[0], _mm_setzero_si128())
			),
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4[1], _mm_set1_epi32(0x80000000)), _mm_set1_epi32(1)
				), _mm_cmpneq_epi32(a.mF4[1], _mm_setzero_si128())
			),
			
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4[2], _mm_set1_epi32(0x80000000)), _mm_set1_epi32(1)
				), _mm_cmpneq_epi32(a.mF4[2], _mm_setzero_si128())
			),
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4[3], _mm_set1_epi32(0x80000000)), _mm_set1_epi32(1)
				), _mm_cmpneq_epi32(a.mF4[3], _mm_setzero_si128())
			)
		);
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 16>& rhs)
	{
		int VCL_ALIGN(16) vars[16];
		_mm_store_si128((__m128i*) (vars +  0), rhs.mF4[0]);
		_mm_store_si128((__m128i*) (vars +  4), rhs.mF4[1]);
		_mm_store_si128((__m128i*) (vars +  8), rhs.mF4[2]);
		_mm_store_si128((__m128i*) (vars + 12), rhs.mF4[3]);

		s << "'" << vars[ 0] << "," << vars[ 1] << "," << vars[ 2] << "," << vars[ 3]
				 << vars[ 4] << "," << vars[ 5] << "," << vars[ 6] << "," << vars[ 7]
				 << vars[ 8] << "," << vars[ 9] << "," << vars[10] << "," << vars[11]
				 << vars[12] << "," << vars[13] << "," << vars[14] << "," << vars[15] << "'";

		return s;
	}
}
