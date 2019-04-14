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
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/bool8_neon.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

namespace Vcl
{
	template<>
	class alignas(16) VectorScalar<int, 8> : protected Core::Simd::VectorScalarBase<int, 8, Core::Simd::SimdExt::NEON>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(NEON)

	public:
		VCL_STRONG_INLINE VectorScalar<int, 8> operator+ (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				vaddq_s32(get(0), rhs.get(0)),
				vaddq_s32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 8> operator- (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				vsubq_s32(get(0), rhs.get(0)),
				vsubq_s32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 8> operator* (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				vmulq_s32(get(0), rhs.get(0)),
				vmulq_s32(get(1), rhs.get(1))
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 8> abs() const
		{
			return VectorScalar<int, 8>
			(
				vabsq_s32(get(0)),
				vabsq_s32(get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 8> max(const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				vmaxq_s32(get(0), rhs.get(0)),
				vmaxq_s32(get(1), rhs.get(1))
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 8> operator& (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				vandq_s32(get(0), rhs.get(0)),
				vandq_s32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 8> operator| (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<int, 8>
			(
				vorrq_s32(get(0), rhs.get(0)),
				vorrq_s32(get(1), rhs.get(1))
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator== (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vceqq_s32(get(0), rhs.get(0)),
				vceqq_s32(get(1), rhs.get(1))
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 8> operator< (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcltq_s32(get(0), rhs.get(0)),
				vcltq_s32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator<= (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcleq_s32(get(0), rhs.get(0)),
				vcleq_s32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator> (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcgtq_s32(get(0), rhs.get(0)),
				vcgtq_s32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator>= (const VectorScalar<int, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcgeq_s32(get(0), rhs.get(0)),
				vcgeq_s32(get(1), rhs.get(1))
			);
		}
	};
	
	VCL_STRONG_INLINE VectorScalar<int, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<int, 8>& a, const VectorScalar<int, 8>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 8>
		(
			vbslq_s32(mask.get(0), a.get(0), b.get(0)),
			vbslq_s32(mask.get(1), a.get(1), b.get(1))
		);
	}

	/*VCL_STRONG_INLINE VectorScalar<int, 8> signum(const VectorScalar<int, 8>& a)
	{
		return VectorScalar<int, 8>
		(
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.get(0), vdupq_n_s32(0x80000000)), vdupq_n_s32(1)
				), _mm_cmpneq_epi32(a.get(0), _mm_setzero_si128())
			),
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.get(1), vdupq_n_s32(0x80000000)), vdupq_n_s32(1)
				), _mm_cmpneq_epi32(a.get(1), _mm_setzero_si128())
			)
		);
	}*/

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 8>& rhs)
	{
		alignas(16) int vars[8];
		vst1q_s32(vars + 0, rhs.get(0));
		vst1q_s32(vars + 4, rhs.get(1));

		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3]
				 << vars[4] << "," << vars[5] << "," << vars[6] << "," << vars[7] << "'";

		return s;
	}
}
