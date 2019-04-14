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
	class alignas(16) VectorScalar<float, 8> : protected Core::Simd::VectorScalarBase<float, 8, Core::Simd::SimdExt::NEON>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(NEON)

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

		VCL_STRONG_INLINE float dot(const VectorScalar<float, 8>& rhs) const
		{
			return
				vdotq_f32(get(0), rhs.get(0)) +
				vdotq_f32(get(1), rhs.get(1));
		}

		VCL_STRONG_INLINE float min() const
		{
			return std::min(vpminq_f32(get(0)), vpminq_f32(get(1)));
		}
		VCL_STRONG_INLINE float max() const
		{
			return std::max(vpmaxq_f32(get(0)), vpmaxq_f32(get(1)));
		}
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
			vbslq_f32(mask.get(0), a.get(0), b.get(0)),
			vbslq_f32(mask.get(1), a.get(1), b.get(1))
		);
	}

	VCL_STRONG_INLINE VectorScalar<bool, 8> isinf(const VectorScalar<float, 8>& x)
	{
		return VectorScalar<bool, 8>
		(
			visinfq_f32(x.get(0)),
			visinfq_f32(x.get(1))
		);
	}
}
