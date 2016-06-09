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
#include <vcl/core/simd/bool16_neon.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

namespace Vcl
{
	template<>
	class VectorScalar<int, 16>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() {}
		VCL_STRONG_INLINE VectorScalar(int s)
		{
			mF4[0] = vdupq_n_s32(s);
			mF4[1] = vdupq_n_s32(s);
			mF4[2] = vdupq_n_s32(s);
			mF4[3] = vdupq_n_s32(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar
		(
			int s00, int s01, int s02, int s03, int s04, int s05, int s06, int s07,
			int s08, int s09, int s10, int s11, int s12, int s13, int s14, int s15
		)
		{
			int VCL_ALIGN(16) d0[4] = { s03, s02, s01, s00 };
			int VCL_ALIGN(16) d1[4] = { s07, s06, s05, s04 };
			int VCL_ALIGN(16) d2[4] = { s11, s10, s09, s08 };
			int VCL_ALIGN(16) d3[4] = { s15, s14, s13, s12 };
			mF4[0] = vld1q_s32(d0);
			mF4[1] = vld1q_s32(d1);
			mF4[1] = vld1q_s32(d2);
			mF4[1] = vld1q_s32(d3);
		}
		VCL_STRONG_INLINE explicit VectorScalar(int32x4_t I4_0, int32x4_t I4_1, int32x4_t I4_2, const int32x4_t& I4_3)
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

			switch (idx % 4)
			{
			case 0:
				return vgetq_lane_f32(get(idx / 4), 0);
			case 1:
				return vgetq_lane_f32(get(idx / 4), 1);
			case 2:
				return vgetq_lane_f32(get(idx / 4), 2);
			case 3:
				return vgetq_lane_f32(get(idx / 4), 3);
			}
		}

		VCL_STRONG_INLINE int32x4_t get(int i) const
		{
			Require(0 <= i && i < 4, "Access is in range.");

			return mF4[i];
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> operator+ (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				vaddq_s32(mF4[0], rhs.mF4[0]),
				vaddq_s32(mF4[1], rhs.mF4[1]),
				vaddq_s32(mF4[2], rhs.mF4[2]),
				vaddq_s32(mF4[3], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator- (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				vsubq_s32(mF4[0], rhs.mF4[0]),
				vsubq_s32(mF4[1], rhs.mF4[1]),
				vsubq_s32(mF4[2], rhs.mF4[2]),
				vsubq_s32(mF4[3], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> operator* (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				vmulq_s32(mF4[0], rhs.mF4[0]),
				vmulq_s32(mF4[1], rhs.mF4[1]),
				vmulq_s32(mF4[2], rhs.mF4[2]),
				vmulq_s32(mF4[3], rhs.mF4[3])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 16> abs()  const
		{
			return VectorScalar<int, 16>
			(
				vabsq_s32(mF4[0]),
				vabsq_s32(mF4[1]),
				vabsq_s32(mF4[2]),
				vabsq_s32(mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<int, 16> max(const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<int, 16>
			(
				vmaxq_s32(mF4[0], rhs.mF4[0]),
				vmaxq_s32(mF4[1], rhs.mF4[1]),
				vmaxq_s32(mF4[2], rhs.mF4[2]),
				vmaxq_s32(mF4[3], rhs.mF4[3])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator== (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vceqq_s32(mF4[0], rhs.mF4[0]),
				vceqq_s32(mF4[1], rhs.mF4[1]),
				vceqq_s32(mF4[0], rhs.mF4[2]),
				vceqq_s32(mF4[1], rhs.mF4[3])
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 16> operator< (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcltq_s32(mF4[0], rhs.mF4[0]),
				vcltq_s32(mF4[1], rhs.mF4[1]),
				vcltq_s32(mF4[0], rhs.mF4[2]),
				vcltq_s32(mF4[1], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator<= (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcleq_s32(mF4[0], rhs.mF4[0]),
				vcleq_s32(mF4[1], rhs.mF4[1]),
				vcleq_s32(mF4[0], rhs.mF4[2]),
				vcleq_s32(mF4[1], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator> (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcgtq_s32(mF4[0], rhs.mF4[0]),
				vcgtq_s32(mF4[1], rhs.mF4[1]),
				vcgtq_s32(mF4[0], rhs.mF4[2]),
				vcgtq_s32(mF4[1], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator>= (const VectorScalar<int, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcgeq_s32(mF4[0], rhs.mF4[0]),
				vcgeq_s32(mF4[1], rhs.mF4[1]),
				vcgeq_s32(mF4[0], rhs.mF4[2]),
				vcgeq_s32(mF4[1], rhs.mF4[3])
			);
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 16>& rhs);
		friend VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b);
		friend VectorScalar<int, 16> signum(const VectorScalar<int, 16>& a);

	private:
		int32x4_t mF4[4];
	};
	
	VCL_STRONG_INLINE VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 16>
		(
			vbslq_s32(mask.mF4[0], a.get(0), b.get(0)),
			vbslq_s32(mask.mF4[1], a.get(1), b.get(1)),
			vbslq_s32(mask.mF4[2], a.get(2), b.get(2)),
			vbslq_s32(mask.mF4[3], a.get(3), b.get(3))
		);
	}

	/*VCL_STRONG_INLINE VectorScalar<int, 16> signum(const VectorScalar<int, 16>& a)
	{
		return VectorScalar<int, 16>
		(
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4[0], vdupq_n_s32(0x80000000)), vdupq_n_s32(1)
				), _mm_cmpneq_epi32(a.mF4[0], _mm_setzero_si128())
			),
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4[1], vdupq_n_s32(0x80000000)), vdupq_n_s32(1)
				), _mm_cmpneq_epi32(a.mF4[1], _mm_setzero_si128())
			),
			
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4[2], vdupq_n_s32(0x80000000)), vdupq_n_s32(1)
				), _mm_cmpneq_epi32(a.mF4[2], _mm_setzero_si128())
			),
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4[3], vdupq_n_s32(0x80000000)), vdupq_n_s32(1)
				), _mm_cmpneq_epi32(a.mF4[3], _mm_setzero_si128())
			)
		);
	}*/

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 16>& rhs)
	{
		int VCL_ALIGN(16) vars[16];
		vst1q_s32(vars +  0, rhs.mF4[0]);
		vst1q_s32(vars +  4, rhs.mF4[1]);
		vst1q_s32(vars +  8, rhs.mF4[2]);
		vst1q_s32(vars + 12, rhs.mF4[3]);

		s << "'" << vars[ 0] << "," << vars[ 1] << "," << vars[ 2] << "," << vars[ 3]
				 << vars[ 4] << "," << vars[ 5] << "," << vars[ 6] << "," << vars[ 7]
				 << vars[ 8] << "," << vars[ 9] << "," << vars[10] << "," << vars[11]
				 << vars[12] << "," << vars[13] << "," << vars[14] << "," << vars[15] << "'";

		return s;
	}
}
