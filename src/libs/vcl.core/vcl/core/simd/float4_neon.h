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

// VCL
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/bool4_neon.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

namespace Vcl
{
	template<>
	class alignas(16) VectorScalar<float, 4> : protected Core::Simd::VectorScalarBase<float, 4, Core::Simd::SimdExt::NEON>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(NEON)

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

		VCL_STRONG_INLINE float dot(const VectorScalar<float, 4>& rhs) const { return vdotq_f32(get(0), rhs.get(0)); }

		VCL_STRONG_INLINE float min() const { return vpminq_f32(get(0)); }
		VCL_STRONG_INLINE float max() const { return vpmaxq_f32(get(0)); }
	};

	VCL_STRONG_INLINE VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b)
	{
		// Straight forward method
		// (b & ~mask) | (a & mask)
		return VectorScalar<float, 4>(vbslq_f32(mask.get(0), a.get(0), b.get(0)));
	}

	VCL_STRONG_INLINE VectorScalar<bool, 4> isinf(const VectorScalar<float, 4>& x)
	{
		return VectorScalar<bool, 4>(visinfq_f32(x.get(0)));
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 4>& rhs)
	{
		alignas(16) float vars[4];
		vst1q_f32(vars + 0, rhs.get(0));
		
		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3] << "'";
		return s;
	}
}
