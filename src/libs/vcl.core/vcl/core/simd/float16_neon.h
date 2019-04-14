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
#include <vcl/core/simd/bool16_neon.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

namespace Vcl
{
	template<>
	class alignas(16) VectorScalar<float, 16> : protected Core::Simd::VectorScalarBase<float, 16, Core::Simd::SimdExt::NEON>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(NEON)

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
				vaddq_f32(get(0), rhs.get(0)),
				vaddq_f32(get(1), rhs.get(1)),
				vaddq_f32(get(2), rhs.get(2)),
				vaddq_f32(get(3), rhs.get(3))
			);
		}

		VectorScalar<float, 16> operator- (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vsubq_f32(get(0), rhs.get(0)),
				vsubq_f32(get(1), rhs.get(1)),
				vsubq_f32(get(2), rhs.get(2)),
				vsubq_f32(get(3), rhs.get(3))
			);
		}

		VectorScalar<float, 16> operator* (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vmulq_f32(get(0), rhs.get(0)),
				vmulq_f32(get(1), rhs.get(1)),
				vmulq_f32(get(2), rhs.get(2)),
				vmulq_f32(get(3), rhs.get(3))
			);
		}

		VectorScalar<float, 16> operator/ (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vdivq_f32(get(0), rhs.get(0)),
				vdivq_f32(get(1), rhs.get(1)),
				vdivq_f32(get(2), rhs.get(2)),
				vdivq_f32(get(3), rhs.get(3))
			);
		}

	public:
		VectorScalar<float, 16>& operator += (const VectorScalar<float, 16>& rhs)
		{
			_data[0] = vaddq_f32(get(0), rhs.get(0));
			_data[1] = vaddq_f32(get(1), rhs.get(1));
			_data[2] = vaddq_f32(get(2), rhs.get(2));
			_data[3] = vaddq_f32(get(3), rhs.get(3));
			return *this;
		}
		VectorScalar<float, 16>& operator -= (const VectorScalar<float, 16>& rhs)
		{
			_data[0] = vsubq_f32(get(0), rhs.get(0));
			_data[1] = vsubq_f32(get(1), rhs.get(1));
			_data[2] = vsubq_f32(get(2), rhs.get(2));
			_data[3] = vsubq_f32(get(3), rhs.get(3));
			return *this;
		}
		VectorScalar<float, 16>& operator *= (const VectorScalar<float, 16>& rhs)
		{
			_data[0] = vmulq_f32(get(0), rhs.get(0));
			_data[1] = vmulq_f32(get(1), rhs.get(1));
			_data[2] = vmulq_f32(get(2), rhs.get(2));
			_data[3] = vmulq_f32(get(3), rhs.get(3));
			return *this;
		}
		VectorScalar<float, 16>& operator /= (const VectorScalar<float, 16>& rhs)
		{
			_data[0] = vdivq_f32(get(0), rhs.get(0));
			_data[1] = vdivq_f32(get(1), rhs.get(1));
			_data[2] = vdivq_f32(get(2), rhs.get(2));
			_data[3] = vdivq_f32(get(3), rhs.get(3));
			return *this;
		}

	public:
		VectorScalar<bool, 16> operator== (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vceqq_f32(get(0), rhs.get(0)),
				vceqq_f32(get(1), rhs.get(1)),
				vceqq_f32(get(2), rhs.get(2)),
				vceqq_f32(get(3), rhs.get(3))
			);
		}
		
		VectorScalar<bool, 16> operator!= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcneqq_f32(get(0), rhs.get(0)),
				vcneqq_f32(get(1), rhs.get(1)),
				vcneqq_f32(get(2), rhs.get(2)),
				vcneqq_f32(get(3), rhs.get(3))
			);
		}
		VectorScalar<bool, 16> operator< (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcltq_f32(get(0), rhs.get(0)),
				vcltq_f32(get(1), rhs.get(1)),
				vcltq_f32(get(2), rhs.get(2)),
				vcltq_f32(get(3), rhs.get(3))
			);
		}
		VectorScalar<bool, 16> operator<= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcleq_f32(get(0), rhs.get(0)),
				vcleq_f32(get(1), rhs.get(1)),
				vcleq_f32(get(2), rhs.get(2)),
				vcleq_f32(get(3), rhs.get(3))
			);
		}
		VectorScalar<bool, 16> operator> (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcgtq_f32(get(0), rhs.get(0)),
				vcgtq_f32(get(1), rhs.get(1)),
				vcgtq_f32(get(2), rhs.get(2)),
				vcgtq_f32(get(3), rhs.get(3))
			);
		}
		VectorScalar<bool, 16> operator>= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				vcgeq_f32(get(0), rhs.get(0)),
				vcgeq_f32(get(1), rhs.get(1)),
				vcgeq_f32(get(2), rhs.get(2)),
				vcgeq_f32(get(3), rhs.get(3))
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> abs()   const { return VectorScalar<float, 16>(vabsq_f32  (get(0)), vabsq_f32  (get(1)), vabsq_f32  (get(2)), vabsq_f32  (get(3))); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sin()   const { return VectorScalar<float, 16>(vsinq_f32  (get(0)), vsinq_f32  (get(1)), vsinq_f32  (get(2)), vsinq_f32  (get(3))); }
		VCL_STRONG_INLINE VectorScalar<float, 16> cos()   const { return VectorScalar<float, 16>(vcosq_f32  (get(0)), vcosq_f32  (get(1)), vcosq_f32  (get(2)), vcosq_f32  (get(3))); }
		VCL_STRONG_INLINE VectorScalar<float, 16> exp()   const { return VectorScalar<float, 16>(vexpq_f32  (get(0)), vexpq_f32  (get(1)), vexpq_f32  (get(2)), vexpq_f32  (get(3))); }
		VCL_STRONG_INLINE VectorScalar<float, 16> log()   const { return VectorScalar<float, 16>(vlogq_f32  (get(0)), vlogq_f32  (get(1)), vlogq_f32  (get(2)), vlogq_f32  (get(3))); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sgn()   const { return VectorScalar<float, 16>(vsgnq_f32  (get(0)), vsgnq_f32  (get(1)), vsgnq_f32  (get(2)), vsgnq_f32  (get(3))); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sqrt()  const { return VectorScalar<float, 16>(vsqrtq_f32 (get(0)), vsqrtq_f32 (get(1)), vsqrtq_f32 (get(2)), vsqrtq_f32 (get(3))); }
		VCL_STRONG_INLINE VectorScalar<float, 16> rcp()   const { return VectorScalar<float, 16>(vrcpq_f32  (get(0)), vrcpq_f32  (get(1)), vrcpq_f32  (get(2)), vrcpq_f32  (get(3))); }
		VCL_STRONG_INLINE VectorScalar<float, 16> rsqrt() const { return VectorScalar<float, 16>(vrsqrtq_f32(get(0)), vrsqrtq_f32(get(1)), vrsqrtq_f32(get(2)), vrsqrtq_f32(get(3))); }

		VCL_STRONG_INLINE VectorScalar<float, 16> acos() const { return VectorScalar<float, 16>(vacosq_f32(get(0)), vacosq_f32(get(1)), vacosq_f32(get(2)), vacosq_f32(get(3))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> min(const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vminq_f32(get(0), rhs.get(0)),
				vminq_f32(get(1), rhs.get(1)),
				vminq_f32(get(2), rhs.get(2)),
				vminq_f32(get(3), rhs.get(3))
			);
		}
		VCL_STRONG_INLINE VectorScalar<float, 16> max(const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				vmaxq_f32(get(0), rhs.get(0)),
				vmaxq_f32(get(1), rhs.get(1)),
				vmaxq_f32(get(2), rhs.get(2)),
				vmaxq_f32(get(3), rhs.get(3))
			);
		}

		VCL_STRONG_INLINE float dot(const VectorScalar<float, 16>& rhs) const
		{
			return
				vdotq_f32(get(0), rhs.get(0)) +
				vdotq_f32(get(1), rhs.get(1)) +
				vdotq_f32(get(2), rhs.get(2)) +
				vdotq_f32(get(3), rhs.get(3));
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
	};

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 16>& rhs)
	{
		alignas(16) float vars[16];
		vst1q_f32(vars +  0, rhs.get(0));
		vst1q_f32(vars +  4, rhs.get(1));
		vst1q_f32(vars +  8, rhs.get(2));
		vst1q_f32(vars + 12, rhs.get(3));

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
			vbslq_f32(mask.get(0), a.get(0), b.get(0)),
			vbslq_f32(mask.get(1), a.get(1), b.get(1)),
			vbslq_f32(mask.get(2), a.get(2), b.get(2)),
			vbslq_f32(mask.get(3), a.get(3), b.get(3))
		);
	}

	VCL_STRONG_INLINE VectorScalar<bool, 16> isinf(const VectorScalar<float, 16>& x)
	{
		return VectorScalar<bool, 16>
		(
			visinfq_f32(x.get(0)),
			visinfq_f32(x.get(1)),
			visinfq_f32(x.get(2)),
			visinfq_f32(x.get(3))
		);
	}
}
