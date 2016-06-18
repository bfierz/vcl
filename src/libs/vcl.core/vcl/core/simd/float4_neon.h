/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
//#include <vcl/core/simd/bool4_sse.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 4>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(const VectorScalar<float, 4>& rhs)
		{
			set(rhs.get(0));
		}
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			set(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(float s0, float s1, float s2, float s3)
		{
			set(s0, s1, s2, s3);
		}
		explicit VCL_STRONG_INLINE VectorScalar(float32x4_t F4)
		{
			set(F4);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator= (const VectorScalar<float, 4>& rhs) { set(rhs.get(0)); return *this; }

	public:
		VCL_STRONG_INLINE float operator[] (int idx) const
		{
			Require(0 <= idx && idx < 4, "Access is in range.");

			switch (idx)
			{
			case 0:
				return vgetq_lane_f32(get(0), 0);
			case 1:
				return vgetq_lane_f32(get(0), 1);
			case 2:
				return vgetq_lane_f32(get(0), 2);
			case 3:
				return vgetq_lane_f32(get(0), 3);
			}
		}

		VCL_STRONG_INLINE float32x4_t get(int i = 0) const
		{
			Require(0 == i, "Access is in range.");

			return _data[i];
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> operator- () const
		{
			return VectorScalar<float, 4>(vnegq_f32(get(0)));
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> operator+ (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(vaddq_f32(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator- (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(vsubq_f32(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator* (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(vmulq_f32(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator/ (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(vdivq_f32(get(0), rhs.get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator += (const VectorScalar<float, 4>& rhs)
		{
			set(vaddq_f32(get(0), rhs.get(0)));
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator -= (const VectorScalar<float, 4>& rhs)
		{
			set(vsubq_f32(get(0), rhs.get(0)));
			return *this;
		}

		VCL_STRONG_INLINE VectorScalar<float, 4>& operator *= (const VectorScalar<float, 4>& rhs)
		{
			set(vmulq_f32(get(0), rhs.get(0)));
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator /= (const VectorScalar<float, 4>& rhs)
		{
			set(vdivq_f32(get(0), rhs.get(0)));
			return *this;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator== (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(vceqq_f32 (get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator!= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(vcneqq_f32(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator<  (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(vcltq_f32 (get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator<= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(vcleq_f32 (get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator>  (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(vcgtq_f32 (get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator>= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(vcgeq_f32 (get(0), rhs.get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> abs()   const { return VectorScalar<float, 4>(vabsq_f32  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sin()   const { return VectorScalar<float, 4>(vsinq_f32  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> cos()   const { return VectorScalar<float, 4>(vcosq_f32  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> exp()   const { return VectorScalar<float, 4>(vexpq_f32  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> log()   const { return VectorScalar<float, 4>(vlogq_f32  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sgn()   const { return VectorScalar<float, 4>(vsgnq_f32  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sqrt()  const { return VectorScalar<float, 4>(vsqrtq_f32 (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> rcp()   const { return VectorScalar<float, 4>(vrcpq_f32  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> rsqrt() const { return VectorScalar<float, 4>(vrsqrtq_f32(get(0))); }

		VCL_STRONG_INLINE VectorScalar<float, 4> acos() const { return VectorScalar<float, 4>(vacosq_f32(get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> min(const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(vminq_f32(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> max(const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(vmaxq_f32(get(0), rhs.get(0))); }

		VCL_STRONG_INLINE float min() const { return vpminq_f32(get(0)); }
		VCL_STRONG_INLINE float max() const { return vpmaxq_f32(get(0)); }

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 4>& rhs);
		friend VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b);
		
	private:
		VCL_STRONG_INLINE void set(float s0)
		{
			_data[0] = vdupq_n_f32(s0);
		}
		VCL_STRONG_INLINE void set(float s0, float s1, float s2, float s3)
		{
			float VCL_ALIGN(16) data[4] = { s0, s1, s2, s3 };
			_data[0] = vld1q_f32(data);
		}
		VCL_STRONG_INLINE void set(float32x4_t vec)
		{
			_data[0] = vec;
		}

	private:
		float32x4_t _data[1];
	};

	VCL_STRONG_INLINE VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b)
	{
		// Straight forward method
		// (b & ~mask) | (a & mask)
		return VectorScalar<float, 4>(vbslq_f32(mask._data[0], a.get(0), b.get(0)));
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 4>& rhs)
	{
		VCL_ALIGN(16) float vars[4];
		vst1q_f32(vars + 0, rhs.get(0));
		
		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3] << "'";
		return s;
	}
}
