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
#include <vcl/config/eigen.h>

// C++ Standard Library
#include <array>

// VCL 
#include <vcl/core/simd/bool16_neon.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 16>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			set(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar
		(
			float s00, float s01, float s02, float s03, float s04, float s05, float s06, float s07,
			float s08, float s09, float s10, float s11, float s12, float s13, float s14, float s15
		)
		{
			set(s00, s01, s02, s03, s04, s05, s06, s07,
				s08, s09, s10, s11, s12, s13, s14, s15);
		}
		explicit VCL_STRONG_INLINE VectorScalar(const float32x4_t& F4_0, const float32x4_t& F4_1, const float32x4_t& F4_2, const float32x4_t& F4_3)
		{
			set(F4_0, F4_1, F4_2, F4_3);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator= (const VectorScalar<float, 16>& rhs)
		{
			mF4[0] = rhs.mF4[0];
			mF4[1] = rhs.mF4[1];
			mF4[2] = rhs.mF4[2];
			mF4[3] = rhs.mF4[3];
			
			return *this;
		}

	public:
		VCL_STRONG_INLINE float operator[] (int idx) const
		{
			VclRequire(0 <= idx && idx < 16, "Access is in range.");

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

		VCL_STRONG_INLINE float32x4_t get(int i) const
		{
			VclRequire(0 <= i && i < 4, "Access is in range.");

			return mF4[i];
		}

	public:
		VectorScalar<float, 16> operator- () const
		{
			return (*this) * VectorScalar<float, 16>(-1);
		}

	public:
		VectorScalar<float, 16> operator+ (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vaddq_f32(mF4[0], rhs.mF4[0]),
				vaddq_f32(mF4[1], rhs.mF4[1]),
				vaddq_f32(mF4[2], rhs.mF4[2]),
				vaddq_f32(mF4[3], rhs.mF4[3])
			);
		}

		VectorScalar<float, 16> operator- (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vsubq_f32(mF4[0], rhs.mF4[0]),
				vsubq_f32(mF4[1], rhs.mF4[1]),
				vsubq_f32(mF4[2], rhs.mF4[2]),
				vsubq_f32(mF4[3], rhs.mF4[3])
			);
		}

		VectorScalar<float, 16> operator* (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vmulq_f32(mF4[0], rhs.mF4[0]),
				vmulq_f32(mF4[1], rhs.mF4[1]),
				vmulq_f32(mF4[2], rhs.mF4[2]),
				vmulq_f32(mF4[3], rhs.mF4[3])
			);
		}

		VectorScalar<float, 16> operator/ (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vdivq_f32(mF4[0], rhs.mF4[0]),
				vdivq_f32(mF4[1], rhs.mF4[1]),
				vdivq_f32(mF4[2], rhs.mF4[2]),
				vdivq_f32(mF4[3], rhs.mF4[3])
			);
		}

	public:
		VectorScalar<float, 16>& operator += (const VectorScalar<float, 16>& rhs)
		{
			mF4[0] = vaddq_f32(mF4[0], rhs.mF4[0]);
			mF4[1] = vaddq_f32(mF4[1], rhs.mF4[1]);
			mF4[2] = vaddq_f32(mF4[2], rhs.mF4[2]);
			mF4[3] = vaddq_f32(mF4[3], rhs.mF4[3]);
			return *this;
		}
		VectorScalar<float, 16>& operator -= (const VectorScalar<float, 16>& rhs)
		{
			mF4[0] = vsubq_f32(mF4[0], rhs.mF4[0]);
			mF4[1] = vsubq_f32(mF4[1], rhs.mF4[1]);
			mF4[2] = vsubq_f32(mF4[2], rhs.mF4[2]);
			mF4[3] = vsubq_f32(mF4[3], rhs.mF4[3]);
			return *this;
		}
		VectorScalar<float, 16>& operator *= (const VectorScalar<float, 16>& rhs)
		{
			mF4[0] = vmulq_f32(mF4[0], rhs.mF4[0]);
			mF4[1] = vmulq_f32(mF4[1], rhs.mF4[1]);
			mF4[2] = vmulq_f32(mF4[2], rhs.mF4[2]);
			mF4[3] = vmulq_f32(mF4[3], rhs.mF4[3]);
			return *this;
		}
		VectorScalar<float, 16>& operator /= (const VectorScalar<float, 16>& rhs)
		{
			mF4[0] = vdivq_f32(mF4[0], rhs.mF4[0]);
			mF4[1] = vdivq_f32(mF4[1], rhs.mF4[1]);
			mF4[2] = vdivq_f32(mF4[2], rhs.mF4[2]);
			mF4[3] = vdivq_f32(mF4[3], rhs.mF4[3]);
			return *this;
		}

	public:
		VectorScalar<bool, 16> operator== (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vceqq_f32(mF4[0], rhs.mF4[0]),
				vceqq_f32(mF4[1], rhs.mF4[1]),
				vceqq_f32(mF4[2], rhs.mF4[2]),
				vceqq_f32(mF4[3], rhs.mF4[3])
			);
		}
		
		VectorScalar<bool, 16> operator!= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcneqq_f32(mF4[0], rhs.mF4[0]),
				vcneqq_f32(mF4[1], rhs.mF4[1]),
				vcneqq_f32(mF4[2], rhs.mF4[2]),
				vcneqq_f32(mF4[3], rhs.mF4[3])
			);
		}
		VectorScalar<bool, 16> operator< (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcltq_f32(mF4[0], rhs.mF4[0]),
				vcltq_f32(mF4[1], rhs.mF4[1]),
				vcltq_f32(mF4[2], rhs.mF4[2]),
				vcltq_f32(mF4[3], rhs.mF4[3])
			);
		}
		VectorScalar<bool, 16> operator<= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcleq_f32(mF4[0], rhs.mF4[0]),
				vcleq_f32(mF4[1], rhs.mF4[1]),
				vcleq_f32(mF4[2], rhs.mF4[2]),
				vcleq_f32(mF4[3], rhs.mF4[3])
			);
		}
		VectorScalar<bool, 16> operator> (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcgtq_f32(mF4[0], rhs.mF4[0]),
				vcgtq_f32(mF4[1], rhs.mF4[1]),
				vcgtq_f32(mF4[2], rhs.mF4[2]),
				vcgtq_f32(mF4[3], rhs.mF4[3])
			);
		}
		VectorScalar<bool, 16> operator>= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcgeq_f32(mF4[0], rhs.mF4[0]),
				vcgeq_f32(mF4[1], rhs.mF4[1]),
				vcgeq_f32(mF4[2], rhs.mF4[2]),
				vcgeq_f32(mF4[3], rhs.mF4[3])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> abs()   const { return VectorScalar<float, 16>(vabsq_f32  (mF4[0]), vabsq_f32  (mF4[1]), vabsq_f32  (mF4[2]), vabsq_f32  (mF4[3])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sin()   const { return VectorScalar<float, 16>(vsinq_f32  (mF4[0]), vsinq_f32  (mF4[1]), vsinq_f32  (mF4[2]), vsinq_f32  (mF4[3])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> cos()   const { return VectorScalar<float, 16>(vcosq_f32  (mF4[0]), vcosq_f32  (mF4[1]), vcosq_f32  (mF4[2]), vcosq_f32  (mF4[3])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> exp()   const { return VectorScalar<float, 16>(vexpq_f32  (mF4[0]), vexpq_f32  (mF4[1]), vexpq_f32  (mF4[2]), vexpq_f32  (mF4[3])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> log()   const { return VectorScalar<float, 16>(vlogq_f32  (mF4[0]), vlogq_f32  (mF4[1]), vlogq_f32  (mF4[2]), vlogq_f32  (mF4[3])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sgn()   const { return VectorScalar<float, 16>(vsgnq_f32  (mF4[0]), vsgnq_f32  (mF4[1]), vsgnq_f32  (mF4[2]), vsgnq_f32  (mF4[3])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sqrt()  const { return VectorScalar<float, 16>(vsqrtq_f32 (mF4[0]), vsqrtq_f32 (mF4[1]), vsqrtq_f32 (mF4[2]), vsqrtq_f32 (mF4[3])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> rcp()   const { return VectorScalar<float, 16>(vrcpq_f32  (mF4[0]), vrcpq_f32  (mF4[1]), vrcpq_f32  (mF4[2]), vrcpq_f32  (mF4[3])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> rsqrt() const { return VectorScalar<float, 16>(vrsqrtq_f32(mF4[0]), vrsqrtq_f32(mF4[1]), vrsqrtq_f32(mF4[2]), vrsqrtq_f32(mF4[3])); }

		VCL_STRONG_INLINE VectorScalar<float, 16> acos() const { return VectorScalar<float, 16>(vacosq_f32(mF4[0]), vacosq_f32(mF4[1]), vacosq_f32(mF4[2]), vacosq_f32(mF4[3])); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> min(const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vminq_f32(mF4[0], rhs.mF4[0]),
				vminq_f32(mF4[1], rhs.mF4[1]),
				vminq_f32(mF4[2], rhs.mF4[2]),
				vminq_f32(mF4[3], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<float, 16> max(const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vmaxq_f32(mF4[0], rhs.mF4[0]),
				vmaxq_f32(mF4[1], rhs.mF4[1]),
				vmaxq_f32(mF4[2], rhs.mF4[2]),
				vmaxq_f32(mF4[3], rhs.mF4[3])
			);
		}

		VCL_STRONG_INLINE float min() const
		{
			return std::min
			(
				std::min(vpminq_f32(get(0)), vpminq_f32(get(1))),
				std::min(vpminq_f32(get(2)), vpminq_f32(get(3)))
			);
		}
		VCL_STRONG_INLINE float max() const
		{
			return std::max
			(
				std::max(vpmaxq_f32(get(0)), vpmaxq_f32(get(1))),
				std::max(vpmaxq_f32(get(2)), vpmaxq_f32(get(3)))
			);
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 16>& rhs);
		friend VectorScalar<float, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<float, 16>& a, const VectorScalar<float, 16>& b);

	private:
		VCL_STRONG_INLINE void set(float s0)
		{
			mF4[0] = vdupq_n_f32(s0);
			mF4[1] = vdupq_n_f32(s0);
			mF4[2] = vdupq_n_f32(s0);
			mF4[3] = vdupq_n_f32(s0);
		}
		VCL_STRONG_INLINE void set(float s00, float s01, float s02, float s03, float s04, float s05, float s06, float s07,
			                       float s08, float s09, float s10, float s11, float s12, float s13, float s14, float s15)
		{
			float VCL_ALIGN(16) d0[4] = { s00, s01, s02, s03 };
			float VCL_ALIGN(16) d1[4] = { s04, s05, s06, s07 };
			float VCL_ALIGN(16) d2[4] = { s08, s09, s10, s11 };
			float VCL_ALIGN(16) d3[4] = { s12, s13, s14, s15 };
			mF4[0] = vld1q_f32(d0);
			mF4[1] = vld1q_f32(d1);
			mF4[2] = vld1q_f32(d2);
			mF4[3] = vld1q_f32(d3);
		}
		VCL_STRONG_INLINE void set(float32x4_t v0, float32x4_t v1, float32x4_t v2, float32x4_t v3)
		{
			mF4[0] = v0;
			mF4[1] = v1;
			mF4[2] = v2;
			mF4[3] = v3;
		}
	private:
		float32x4_t mF4[4];
	};

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 16>& rhs)
	{
		float VCL_ALIGN(16) vars[16];
		vst1q_f32(vars +  0, rhs.mF4[0]);
		vst1q_f32(vars +  4, rhs.mF4[1]);
		vst1q_f32(vars +  8, rhs.mF4[2]);
		vst1q_f32(vars + 12, rhs.mF4[3]);

		s << "'" << vars[ 0] << "," << vars[ 1] << "," << vars[ 2] << "," << vars[ 3]
		         << vars[ 4] << "," << vars[ 5] << "," << vars[ 6] << "," << vars[ 7]
				 << vars[ 8] << "," << vars[ 9] << "," << vars[10] << "," << vars[11]
				 << vars[12] << "," << vars[13] << "," << vars[14] << "," << vars[15] << "'";

		return s;
	}

	VCL_STRONG_INLINE VectorScalar<float, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<float, 16>& a, const VectorScalar<float, 16>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<float, 16>
		(
			vbslq_f32(mask.mF4[0], a.get(0), b.get(0)),
			vbslq_f32(mask.mF4[1], a.get(1), b.get(1)),
			vbslq_f32(mask.mF4[2], a.get(2), b.get(2)),
			vbslq_f32(mask.mF4[3], a.get(3), b.get(3))
		);
	}
}
