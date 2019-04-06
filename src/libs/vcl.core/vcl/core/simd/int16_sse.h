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
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<int, 16> : protected Core::Simd::VectorScalarBase<int, 16, Core::Simd::SimdExt::SSE>
	{
	public:
		using Core::Simd::VectorScalarBase<int, 16, Core::Simd::SimdExt::SSE>::operator[];
		using Core::Simd::VectorScalarBase<int, 16, Core::Simd::SimdExt::SSE>::get;

		VCL_STRONG_INLINE VectorScalar() {}
		VCL_STRONG_INLINE VectorScalar(int s)
		{
			set(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar
		(
			int s00, int s01, int s02, int s03, int s04, int s05, int s06, int s07,
			int s08, int s09, int s10, int s11, int s12, int s13, int s14, int s15
		)
		{
			set(s00, s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s14, s15);
		}
		VCL_STRONG_INLINE explicit VectorScalar(__m128i I4_0, __m128i I4_1, __m128i I4_2, const __m128i& I4_3)
		{
			set(I4_0, I4_1, I4_2, I4_3);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16>& operator = (const VectorScalar<int, 16>& rhs)
		{
			set(rhs.get(0), rhs.get(1), rhs.get(2), rhs.get(3));
			return *this;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> operator+ (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mm_add_epi32(get(0), rhs.get(0)),
				_mm_add_epi32(get(1), rhs.get(1)),
				_mm_add_epi32(get(2), rhs.get(2)),
				_mm_add_epi32(get(3), rhs.get(3))
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator- (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mm_sub_epi32(get(0), rhs.get(0)),
				_mm_sub_epi32(get(1), rhs.get(1)),
				_mm_sub_epi32(get(2), rhs.get(2)),
				_mm_sub_epi32(get(3), rhs.get(3))
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator* (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_mullo_epi32(get(0), rhs.get(0)),
				_mmVCL_mullo_epi32(get(1), rhs.get(1)),
				_mmVCL_mullo_epi32(get(2), rhs.get(2)),
				_mmVCL_mullo_epi32(get(3), rhs.get(3))
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> abs()  const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_abs_epi32(get(0)),
				_mmVCL_abs_epi32(get(1)),
				_mmVCL_abs_epi32(get(2)),
				_mmVCL_abs_epi32(get(3))
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> max(const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				_mmVCL_max_epi32(get(0), rhs.get(0)),
				_mmVCL_max_epi32(get(1), rhs.get(1)),
				_mmVCL_max_epi32(get(2), rhs.get(2)),
				_mmVCL_max_epi32(get(3), rhs.get(3))
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> operator& (const VectorScalar<int, 16>& rhs)
		{
			return VectorScalar<int, 16>
			(
				_mm_and_si128(get(0), rhs.get(0)),
				_mm_and_si128(get(1), rhs.get(1)),
				_mm_and_si128(get(2), rhs.get(2)),
				_mm_and_si128(get(3), rhs.get(3))
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator| (const VectorScalar<int, 16>& rhs)
		{
			return VectorScalar<int, 16>
			(
				_mm_or_si128(get(0), rhs.get(0)),
				_mm_or_si128(get(1), rhs.get(1)),
				_mm_or_si128(get(2), rhs.get(2)),
				_mm_or_si128(get(3), rhs.get(3))
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator== (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmpeq_epi32(get(0), rhs.get(0)),
				_mm_cmpeq_epi32(get(1), rhs.get(1)),
				_mm_cmpeq_epi32(get(2), rhs.get(2)),
				_mm_cmpeq_epi32(get(3), rhs.get(3))
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 16> operator< (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmplt_epi32(get(0), rhs.get(0)),
				_mm_cmplt_epi32(get(1), rhs.get(1)),
				_mm_cmplt_epi32(get(2), rhs.get(2)),
				_mm_cmplt_epi32(get(3), rhs.get(3))
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator<= (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmple_epi32(get(0), rhs.get(0)),
				_mm_cmple_epi32(get(1), rhs.get(1)),
				_mm_cmple_epi32(get(2), rhs.get(2)),
				_mm_cmple_epi32(get(3), rhs.get(3))
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator> (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmpgt_epi32(get(0), rhs.get(0)),
				_mm_cmpgt_epi32(get(1), rhs.get(1)),
				_mm_cmpgt_epi32(get(2), rhs.get(2)),
				_mm_cmpgt_epi32(get(3), rhs.get(3))
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator>= (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_cmpge_epi32(get(0), rhs.get(0)),
				_mm_cmpge_epi32(get(1), rhs.get(1)),
				_mm_cmpge_epi32(get(2), rhs.get(2)),
				_mm_cmpge_epi32(get(3), rhs.get(3))
			);
		}
	};
	
	VCL_STRONG_INLINE VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 16>
		(
			_mm_xor_si128(b.get(0), _mm_and_si128(_mm_castps_si128(mask.get(0)), _mm_xor_si128(b.get(0), a.get(0)))),
			_mm_xor_si128(b.get(1), _mm_and_si128(_mm_castps_si128(mask.get(1)), _mm_xor_si128(b.get(1), a.get(1)))),
			_mm_xor_si128(b.get(2), _mm_and_si128(_mm_castps_si128(mask.get(2)), _mm_xor_si128(b.get(2), a.get(2)))),
			_mm_xor_si128(b.get(3), _mm_and_si128(_mm_castps_si128(mask.get(3)), _mm_xor_si128(b.get(3), a.get(3))))
		);
	}

	VCL_STRONG_INLINE VectorScalar<int, 16> signum(const VectorScalar<int, 16>& a)
	{
		return VectorScalar<int, 16>
		(
			Core::Simd::SSE::signum(a.get(0)),
			Core::Simd::SSE::signum(a.get(1)),
			Core::Simd::SSE::signum(a.get(2)),
			Core::Simd::SSE::signum(a.get(3))
		);
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 16>& rhs)
	{
		alignas(16) int vars[16];
		_mm_store_si128(reinterpret_cast<__m128i*>(vars +  0), rhs.get(0));
		_mm_store_si128(reinterpret_cast<__m128i*>(vars +  4), rhs.get(1));
		_mm_store_si128(reinterpret_cast<__m128i*>(vars +  8), rhs.get(2));
		_mm_store_si128(reinterpret_cast<__m128i*>(vars + 12), rhs.get(3));

		s << "'" << vars[ 0] << ", " << vars[ 1] << ", " << vars[ 2] << ", " << vars[ 3]
				 << vars[ 4] << ", " << vars[ 5] << ", " << vars[ 6] << ", " << vars[ 7]
				 << vars[ 8] << ", " << vars[ 9] << ", " << vars[10] << ", " << vars[11]
				 << vars[12] << ", " << vars[13] << ", " << vars[14] << ", " << vars[15] << "'";

		return s;
	}
}
