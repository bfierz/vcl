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
#include <vcl/core/simd/bool4_sse.h>
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<int, 4> : protected Core::Simd::VectorScalarBase<int, 4, Core::Simd::SimdExt::SSE>
	{
	public:
		using Core::Simd::VectorScalarBase<int, 4, Core::Simd::SimdExt::SSE>::operator[];
		using Core::Simd::VectorScalarBase<int, 4, Core::Simd::SimdExt::SSE>::get;

		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(int s)
		{
			set(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(int s0, int s1, int s2, int s3)
		{
			set(s0, s1, s2, s3);
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m128i F4)
		{
			set(F4);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 4>& operator= (const VectorScalar<int, 4>& rhs) { set(rhs.get(0)); return *this; }

	public:
		VCL_STRONG_INLINE VectorScalar<int, 4> operator+ (const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mm_add_epi32(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<int, 4> operator- (const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mm_sub_epi32(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<int, 4> operator* (const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mmVCL_mullo_epi32(get(0), rhs.get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<int, 4> abs() const { return VectorScalar<int, 4>(_mmVCL_abs_epi32(get(0))); }
		VCL_STRONG_INLINE VectorScalar<int, 4> max(const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mmVCL_max_epi32(get(0), rhs.get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<int, 4> operator& (const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mm_and_si128(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<int, 4> operator| (const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mm_or_si128(get(0), rhs.get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator== (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmpeq_epi32(get(0), rhs.get(0)));
		}

		VCL_STRONG_INLINE VectorScalar<bool, 4> operator< (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmplt_epi32(get(0), rhs.get(0)));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator<= (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmple_epi32(get(0), rhs.get(0)));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator> (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmpgt_epi32(get(0), rhs.get(0)));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator>= (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmpge_epi32(get(0), rhs.get(0)));
		}
	};

	VCL_STRONG_INLINE VectorScalar<int, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<int, 4>& a, const VectorScalar<int, 4>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 4>(_mm_xor_si128(b.get(0), _mm_and_si128(_mm_castps_si128(mask.get(0)), _mm_xor_si128(b.get(0), a.get(0)))));
	}

	VCL_STRONG_INLINE VectorScalar<int, 4> signum(const VectorScalar<int, 4>& a)
	{
		return VectorScalar<int, 4>(Core::Simd::SSE::signum(a.get(0)));
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 4>& rhs)
	{
		alignas(16) int vars[4];
		_mm_store_si128(reinterpret_cast<__m128i*>(vars + 0), rhs.get(0));

		s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3] << "'";

		return s;
	}
}
