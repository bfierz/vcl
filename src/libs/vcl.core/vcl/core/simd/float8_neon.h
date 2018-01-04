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
#include <vcl/core/simd/bool8_neon.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 8>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			set(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7)
		{
			set(s0, s1, s2, s3, s4, s5, s6, s7);
		}
		VCL_STRONG_INLINE explicit VectorScalar(float32x4_t F4_0, float32x4_t F4_1)
		{
			set(F4_0, F4_1);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator = (const VectorScalar<float, 8>& rhs)
		{
			set(rhs.get(0), rhs.get(1));

			return *this;
		}

	public:
		VCL_STRONG_INLINE float operator[] (int idx) const
		{
			VclRequire(0 <= idx && idx < 8, "Access is in range.");

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
			VclRequire(0 <= i && i < 2, "Access is in range.");

			return _data[i];
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8> operator- () const
		{
			return (*this) * VectorScalar<float, 8>(-1);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8> operator+ (const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(vaddq_f32(get(0), rhs.get(0)), vaddq_f32(get(1), rhs.get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> operator- (const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(vsubq_f32(get(0), rhs.get(0)), vsubq_f32(get(1), rhs.get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> operator* (const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(vmulq_f32(get(0), rhs.get(0)), vmulq_f32(get(1), rhs.get(1))); }

		VCL_STRONG_INLINE VectorScalar<float, 8> operator/ (const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(vdivq_f32(get(0), rhs.get(0)), vdivq_f32(get(1), rhs.get(1))); }
		
	public:
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator += (const VectorScalar<float, 8>& rhs)
		{
			set
			(
				vaddq_f32(get(0), rhs.get(0)),
				vaddq_f32(get(1), rhs.get(1))
			);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator -= (const VectorScalar<float, 8>& rhs)
		{
			set
			(
				vsubq_f32(get(0), rhs.get(0)),
				vsubq_f32(get(1), rhs.get(1))
			);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator *= (const VectorScalar<float, 8>& rhs)
		{			
			set
			(
				vmulq_f32(get(0), rhs.get(0)),
				vmulq_f32(get(1), rhs.get(1))
			);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator /= (const VectorScalar<float, 8>& rhs)
		{			
			set
			(
				vdivq_f32(get(0), rhs.get(0)),
				vdivq_f32(get(1), rhs.get(1))
			);
			return *this;
		}
		
	public:
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator== (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vceqq_f32(get(0), rhs.get(0)),
				vceqq_f32(get(1), rhs.get(1))
			);
		}
		
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator!= (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcneqq_f32(get(0), rhs.get(0)),
				vcneqq_f32(get(1), rhs.get(1))
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 8> operator< (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcltq_f32(get(0), rhs.get(0)),
				vcltq_f32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator<= (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcleq_f32(get(0), rhs.get(0)),
				vcleq_f32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE 	VectorScalar<bool, 8> operator> (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcgtq_f32(get(0), rhs.get(0)),
				vcgtq_f32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator>= (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>
			(
				vcgeq_f32(get(0), rhs.get(0)),
				vcgeq_f32(get(1), rhs.get(1))
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8> abs()   const { return VectorScalar<float, 8>(vabsq_f32  (get(0)), vabsq_f32  (get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> sin()   const { return VectorScalar<float, 8>(vsinq_f32  (get(0)), vsinq_f32  (get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> cos()   const { return VectorScalar<float, 8>(vcosq_f32  (get(0)), vcosq_f32  (get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> exp()   const { return VectorScalar<float, 8>(vexpq_f32  (get(0)), vexpq_f32  (get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> log()   const { return VectorScalar<float, 8>(vlogq_f32  (get(0)), vlogq_f32  (get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> sgn()   const { return VectorScalar<float, 8>(vsgnq_f32  (get(0)), vsgnq_f32  (get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> sqrt()  const { return VectorScalar<float, 8>(vsqrtq_f32 (get(0)), vsqrtq_f32 (get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> rcp()   const { return VectorScalar<float, 8>(vrcpq_f32  (get(0)), vrcpq_f32  (get(1))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> rsqrt() const { return VectorScalar<float, 8>(vrsqrtq_f32(get(0)), vrsqrtq_f32(get(1))); }

		VCL_STRONG_INLINE VectorScalar<float, 8> acos() const { return VectorScalar<float, 8>(vacosq_f32(get(0)), vacosq_f32(get(1))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8> min(const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<float, 8>
			(
				vminq_f32(get(0), rhs.get(0)),
				vminq_f32(get(1), rhs.get(1))
			);
		}
		VCL_STRONG_INLINE VectorScalar<float, 8> max(const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<float, 8>
			(
				vmaxq_f32(get(0), rhs.get(0)),
				vmaxq_f32(get(1), rhs.get(1))
			);
		}

		VCL_STRONG_INLINE float min() const
		{
			return std::min(vpminq_f32(get(0)), vpminq_f32(get(1)));
		}
		VCL_STRONG_INLINE float max() const
		{
			return std::min(vpmaxq_f32(get(0)), vpmaxq_f32(get(1)));
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 8>& rhs);
		friend VectorScalar<float, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<float, 8>& a, const VectorScalar<float, 8>& b);

	private:
		VCL_STRONG_INLINE void set(float s0)
		{
			_data[0] = vdupq_n_f32(s0);
			_data[1] = vdupq_n_f32(s0);
		}
		VCL_STRONG_INLINE void set(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7)
		{
			float alignas(16) d0[4] = { s0, s1, s2, s3 };
			float alignas(16) d1[4] = { s4, s5, s6, s7 };
			_data[0] = vld1q_f32(d0);
			_data[1] = vld1q_f32(d1);
		}
		VCL_STRONG_INLINE void set(float32x4_t v0, float32x4_t v1)
		{
			_data[0] = v0;
			_data[1] = v1;
		}
	private:
		float32x4_t _data[2];
	};

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 8>& rhs)
	{
		alignas(16) float vars[8];
		vst1q_f32(vars + 0, rhs.get(0));
		vst1q_f32(vars + 4, rhs.get(1));
		
		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3]
				 << vars[4] << "," << vars[5] << "," << vars[6] << "," << vars[7] << "'";

		return s;
	}

	VCL_STRONG_INLINE VectorScalar<float, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<float, 8>& a, const VectorScalar<float, 8>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<float, 8>
		(
			vbslq_f32(mask.mF4[0], a.get(0), b.get(0)),
			vbslq_f32(mask.mF4[1], a.get(1), b.get(1))
		);
	}
}
