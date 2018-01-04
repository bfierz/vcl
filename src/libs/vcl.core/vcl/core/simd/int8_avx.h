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
#include <vcl/core/simd/bool8_avx.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_avx.h>

namespace Vcl
{
	template<>
	class VectorScalar<int, 8>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() {}
		VCL_STRONG_INLINE VectorScalar(int s)
		{
			mF8 = _mm256_set1_epi32(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar
		(
			int s00, int s01, int s02, int s03, int s04, int s05, int s06, int s07
		)
		{
			mF8 = _mm256_set_epi32(s07, s06, s05, s04, s03, s02, s01, s00);
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m256i I8) : mF8(I8) {}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 8>& operator= (const VectorScalar<int, 8>& rhs) { mF8 = rhs.mF8; return *this; }

	public:
		VCL_STRONG_INLINE int operator[] (int idx) const
		{
			VclRequire(0 <= idx && idx < 8, "Access is in range.");

			return _mmVCL_extract_epi32(mF8, idx);
		}

		VCL_STRONG_INLINE explicit operator __m256i() const
		{
			return mF8;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 8> operator+ (const VectorScalar<int, 8>& rhs) const { return VectorScalar<int, 8>(_mmVCL_add_epi32(mF8, rhs.mF8)); }
		VCL_STRONG_INLINE VectorScalar<int, 8> operator- (const VectorScalar<int, 8>& rhs) const { return VectorScalar<int, 8>(_mmVCL_sub_epi32(mF8, rhs.mF8)); }
		VCL_STRONG_INLINE VectorScalar<int, 8> operator* (const VectorScalar<int, 8>& rhs) const { return VectorScalar<int, 8>(_mmVCL_mullo_epi32(mF8, rhs.mF8)); }

		VCL_STRONG_INLINE VectorScalar<int, 8>& operator+= (const VectorScalar<int, 8>& rhs) { 
			mF8 = _mmVCL_add_epi32(mF8, rhs.mF8);
			return *this;
		}

		VCL_STRONG_INLINE VectorScalar<int, 8>& operator-= (const VectorScalar<int, 8>& rhs) { 
			mF8 = _mmVCL_sub_epi32(mF8, rhs.mF8);
			return *this;
		}

		VCL_STRONG_INLINE VectorScalar<int, 8>& operator*= (const VectorScalar<int, 8>& rhs) { 
			mF8 = _mmVCL_mullo_epi32(mF8, rhs.mF8);
			return *this;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 8> abs() const
		{
			return VectorScalar<int, 8>
			(
				_mmVCL_abs_epi32(mF8)
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 8> max(const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				_mmVCL_max_epi32(mF8, rhs.mF8)
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator== (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mmVCL_cmpeq_epi32(mF8, rhs.mF8));
		}

		VCL_STRONG_INLINE VectorScalar<bool, 8> operator< (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mmVCL_cmplt_epi32(mF8, rhs.mF8));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator<= (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mmVCL_cmple_epi32(mF8, rhs.mF8));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator> (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mmVCL_cmpgt_epi32(mF8, rhs.mF8));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator>= (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mmVCL_cmpge_epi32(mF8, rhs.mF8));
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 8>& rhs);
		friend VectorScalar<int, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<int, 8>& a, const VectorScalar<int, 8>& b);
		friend VectorScalar<int, 8> signum(const VectorScalar<int, 8>& a);

	private:
		__m256i mF8;
	};

	VCL_STRONG_INLINE VectorScalar<int, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<int, 8>& a, const VectorScalar<int, 8>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 8>(_mmVCL_xor_si256(b.mF8, _mmVCL_and_si256(_mm256_castps_si256(mask.mF8), _mmVCL_xor_si256(b.mF8, a.mF8))));
	}

	VCL_STRONG_INLINE VectorScalar<int, 8> signum(const VectorScalar<int, 8>& a)
	{
		return VectorScalar<int, 8>
		(
			_mmVCL_and_si256
			(
				_mmVCL_or_si256
				(
					_mmVCL_and_si256(a.mF8, _mm256_set1_epi32(0x80000000)), _mm256_set1_epi32(1)
				), _mmVCL_cmpneq_epi32(a.mF8, _mm256_setzero_si256())
			)
		);
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 8>& rhs)
	{
		alignas(32) int vars[8];
		_mm256_store_si256((__m256i*) (vars + 0), rhs.mF8);

		s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3]
			     << vars[4] << ", " << vars[5] << ", " << vars[6] << ", " << vars[7] << "'";

		return s;
	}
}
