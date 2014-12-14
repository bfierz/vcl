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
#include <vcl/core/simd/bool8_sse.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<int, 8>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() {}
		VCL_STRONG_INLINE VectorScalar(int s)
		{
			mF4[0] = _mm_set1_epi32(s);
			mF4[1] = _mm_set1_epi32(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7)
		{
			mF4[0] = _mm_set_epi32(s3, s2, s1, s0);
			mF4[1] = _mm_set_epi32(s7, s6, s5, s4);
		}
		VCL_STRONG_INLINE explicit VectorScalar(__m128i I4_0, __m128i I4_1)
		{
			mF4[0] = I4_0;
			mF4[1] = I4_1;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 8>& operator = (const VectorScalar<int, 8>& rhs)
		{
			mF4[0] = rhs.mF4[0];
			mF4[1] = rhs.mF4[1];

			return *this;
		}

	public:
		VCL_STRONG_INLINE int operator[] (int idx) const
		{
			Require(0 <= idx && idx < 8, "Access is in range.");

			return _mmVCL_extract_epi32(mF4[idx / 4], idx % 4);
		}

		VCL_STRONG_INLINE __m128i get(int i) const
		{
			Require(0 <= i && i < 2, "Access is in range.");

			return mF4[i];
		}
	public:
		VCL_STRONG_INLINE VectorScalar<int, 8> operator+ (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				_mm_add_epi32(mF4[0], rhs.mF4[0]),
				_mm_add_epi32(mF4[1], rhs.mF4[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 8> operator- (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				_mm_sub_epi32(mF4[0], rhs.mF4[0]),
				_mm_sub_epi32(mF4[1], rhs.mF4[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 8> operator* (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				_mmVCL_mullo_epi32(mF4[0], rhs.mF4[0]),
				_mmVCL_mullo_epi32(mF4[1], rhs.mF4[1])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 8> abs() const
		{
			return VectorScalar<int, 8>
			(
				_mm_abs_epi32(mF4[0]),
				_mm_abs_epi32(mF4[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 8> max(const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				_mm_max_epi32(mF4[0], rhs.mF4[0]),
				_mm_max_epi32(mF4[1], rhs.mF4[1])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator== (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				_mm_cmpeq_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmpeq_epi32(mF4[1], rhs.mF4[1])
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 8> operator< (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				_mm_cmplt_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmplt_epi32(mF4[1], rhs.mF4[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator<= (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				_mm_cmple_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmple_epi32(mF4[1], rhs.mF4[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator> (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				_mm_cmpgt_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmpgt_epi32(mF4[1], rhs.mF4[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator>= (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				_mm_cmpge_epi32(mF4[0], rhs.mF4[0]),
				_mm_cmpge_epi32(mF4[1], rhs.mF4[1])
			);
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 8>& rhs);
		friend VectorScalar<int, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<int, 8>& a, const VectorScalar<int, 8>& b);
		friend VectorScalar<int, 8> signum(const VectorScalar<int, 8>& a);

	private:
		std::array<__m128i, 2> mF4;
	};
	
	VCL_STRONG_INLINE VectorScalar<int, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<int, 8>& a, const VectorScalar<int, 8>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 8>
		(
			_mm_xor_si128(b.mF4[0], _mm_and_si128(_mm_castps_si128(mask.mF4[0]), _mm_xor_si128(b.mF4[0], a.mF4[0]))),
			_mm_xor_si128(b.mF4[1], _mm_and_si128(_mm_castps_si128(mask.mF4[1]), _mm_xor_si128(b.mF4[1], a.mF4[1])))
		);
	}

	VCL_STRONG_INLINE VectorScalar<int, 8> signum(const VectorScalar<int, 8>& a)
	{
		return VectorScalar<int, 8>
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
			)
		);
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 8>& rhs)
	{
		int VCL_ALIGN(16) vars[8];
		_mm_store_si128((__m128i*) (vars + 0), rhs.mF4[0]);
		_mm_store_si128((__m128i*) (vars + 4), rhs.mF4[1]);

		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3]
				 << vars[4] << "," << vars[5] << "," << vars[6] << "," << vars[7] << "'";

		return s;
	}
}
