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
#include <vcl/core/simd/bool16_avx.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_avx.h>

namespace Vcl
{
	template<>
	class VectorScalar<int, 16>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() {}
		VCL_STRONG_INLINE VectorScalar(int s)
		{
			mF8[0] = _mm256_set1_epi32(s);
			mF8[1] = _mm256_set1_epi32(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar
		(
			int s00, int s01, int s02, int s03, int s04, int s05, int s06, int s07,
			int s08, int s09, int s10, int s11, int s12, int s13, int s14, int s15
		)
		{
			mF8[0] = _mm256_set_epi32(s07, s06, s05, s04, s03, s02, s01, s00);
			mF8[1] = _mm256_set_epi32(s15, s14, s13, s12, s11, s10, s09, s08);
		}
		VCL_STRONG_INLINE explicit VectorScalar(__m256i I8_0, __m256i I8_1)
		{
			mF8[0] = I8_0;
			mF8[1] = I8_1;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16>& operator = (const VectorScalar<int, 16>& rhs)
		{
			mF8[0] = rhs.mF8[0];
			mF8[1] = rhs.mF8[1];

			return *this;
		}

	public:
		VCL_STRONG_INLINE int operator[] (int idx) const
		{
			Require(0 <= idx && idx < 16, "Access is in range.");

			return _mmVCL_extract_epi32(mF8[idx / 8], idx % 8);
		}

		VCL_STRONG_INLINE __m256i get(int i) const
		{
			Require(0 <= i && i < 2, "Access is in range.");

			return mF8[i];
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> operator+ (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_add_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_add_epi32(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator- (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_sub_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_sub_epi32(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator* (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_mullo_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_mullo_epi32(mF8[1], rhs.mF8[1])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> abs() const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_abs_epi32(mF8[0]),
				_mmVCL_abs_epi32(mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> max(const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_max_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_max_epi32(mF8[1], rhs.mF8[1])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator== (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mmVCL_cmpeq_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_cmpeq_epi32(mF8[1], rhs.mF8[1])
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 16> operator< (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mmVCL_cmplt_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_cmplt_epi32(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator<= (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mmVCL_cmple_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_cmple_epi32(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator> (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mmVCL_cmpgt_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_cmpgt_epi32(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator>= (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mmVCL_cmpge_epi32(mF8[0], rhs.mF8[0]),
				_mmVCL_cmpge_epi32(mF8[1], rhs.mF8[1])
			);
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 16>& rhs);
		friend VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b);
		friend VectorScalar<int, 16> signum(const VectorScalar<int, 16>& a);

	private:
		std::array<__m256i, 2> mF8;
	};
	
	VCL_STRONG_INLINE VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 16>
		(
			_mmVCL_xor_si256(b.mF8[0], _mmVCL_and_si256(_mm256_castps_si256(mask.mF8[0]), _mmVCL_xor_si256(b.mF8[0], a.mF8[0]))),
			_mmVCL_xor_si256(b.mF8[1], _mmVCL_and_si256(_mm256_castps_si256(mask.mF8[1]), _mmVCL_xor_si256(b.mF8[1], a.mF8[1])))
		);
	}

	VCL_STRONG_INLINE VectorScalar<int, 16> signum(const VectorScalar<int, 16>& a)
	{
		return VectorScalar<int, 16>
		(
			_mmVCL_and_si256
			(
				_mmVCL_or_si256
				(
					_mmVCL_and_si256(a.mF8[0], _mm256_set1_epi32(0x80000000)), _mm256_set1_epi32(1)
				), _mmVCL_cmpneq_epi32(a.mF8[0], _mm256_setzero_si256())
			),
			_mmVCL_and_si256
			(
				_mmVCL_or_si256
				(
					_mmVCL_and_si256(a.mF8[1], _mm256_set1_epi32(0x80000000)), _mm256_set1_epi32(1)
				), _mmVCL_cmpneq_epi32(a.mF8[1], _mm256_setzero_si256())
			)
		);
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 16>& rhs)
	{
		int VCL_ALIGN(16) vars[8];
		_mm256_store_si256((__m256i*) (vars + 0), rhs.mF8[0]);
		_mm256_store_si256((__m256i*) (vars + 4), rhs.mF8[1]);

		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3]
				 << vars[4] << "," << vars[5] << "," << vars[6] << "," << vars[7] << "'";

		return s;
	}
}
