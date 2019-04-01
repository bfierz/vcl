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

// VCL 
#include <vcl/core/simd/bool8_avx.h>
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_avx.h>

namespace Vcl
{
	template<>
	class alignas(32) VectorScalar<float, 8> : protected Core::Simd::VectorScalarBase<float, 8, Core::Simd::SimdExt::AVX>
	{
	public:
		using Base = Core::Simd::VectorScalarBase<float, 8, Core::Simd::SimdExt::AVX>;
		
		using Base::operator[];
		using Base::get;

		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(const VectorScalar<float, 8>& rhs)
		{
			_data[0] = rhs.get(0);
		}
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			_data[0] = _mm256_set1_ps(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7)
		{
			_data[0] = _mm256_set_ps(s7, s6, s5, s4, s3, s2, s1, s0);
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m256 F8) { set(F8); }

	public:
		VCL_STRONG_INLINE const VectorScalar<float, 8>& operator= (const VectorScalar<float, 8>& rhs) { _data[0] = rhs.get(0); return *this; }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8> operator+ (const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(_mm256_add_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> operator- (const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(_mm256_sub_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> operator* (const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(_mm256_mul_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> operator/ (const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(_mm256_div_ps(get(0), rhs.get(0))); }
		
	public:
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator += (const VectorScalar<float, 8>& rhs)
		{
			_data[0] = _mm256_add_ps(get(0), rhs.get(0));
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator -= (const VectorScalar<float, 8>& rhs)
		{
			_data[0] = _mm256_sub_ps(get(0), rhs.get(0));
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator *= (const VectorScalar<float, 8>& rhs)
		{
			_data[0] = _mm256_mul_ps(get(0), rhs.get(0));
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator /= (const VectorScalar<float, 8>& rhs)
		{
			_data[0] = _mm256_div_ps(get(0), rhs.get(0));
			return *this;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator== (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mm256_cmp_ps(get(0), rhs.get(0), _CMP_EQ_OQ));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator!= (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mm256_cmp_ps(get(0), rhs.get(0), _CMP_NEQ_OQ));
		}
		
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator< (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mm256_cmp_ps(get(0), rhs.get(0), _CMP_LT_OQ));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator<= (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mm256_cmp_ps(get(0), rhs.get(0), _CMP_LE_OQ));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator> (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mm256_cmp_ps(get(0), rhs.get(0), _CMP_GT_OQ));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator>= (const VectorScalar<float, 8>& rhs) const
		{
			return VectorScalar<bool, 8>(_mm256_cmp_ps(get(0), rhs.get(0), _CMP_GE_OQ));
		}
		
	public:
		VCL_STRONG_INLINE VectorScalar<float, 8> abs()   const { return VectorScalar<float, 8>(_mm256_abs_ps  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> sin()   const { return VectorScalar<float, 8>(_mm256_sin_ps  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> cos()   const { return VectorScalar<float, 8>(_mm256_cos_ps  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> exp()   const { return VectorScalar<float, 8>(_mm256_exp_ps  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> log()   const { return VectorScalar<float, 8>(_mm256_log_ps  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> sgn()   const { return VectorScalar<float, 8>(_mm256_sgn_ps  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> sqrt()  const { return VectorScalar<float, 8>(_mm256_sqrt_ps (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> rcp()   const { return VectorScalar<float, 8>(_mmVCL_rcp_ps  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> rsqrt() const { return VectorScalar<float, 8>(_mmVCL_rsqrt_ps(get(0))); }

		VCL_STRONG_INLINE VectorScalar<float, 8> acos() const { return VectorScalar<float, 8>(_mm256_acos_ps(get(0))); }
		
		VCL_STRONG_INLINE VectorScalar<bool, 8> isinf() const { return VectorScalar<bool, 8>(_mm256_isinf_ps(get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8> min(const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(_mm256_min_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 8> max(const VectorScalar<float, 8>& rhs) const { return VectorScalar<float, 8>(_mm256_max_ps(get(0), rhs.get(0))); }

		VCL_STRONG_INLINE float dot(const VectorScalar<float, 8>& rhs) const { return _mmVCL_dp_ps(get(0), rhs.get(0)); }

		VCL_STRONG_INLINE float min() const { return _mmVCL_hmin_ps(get(0)); }
		VCL_STRONG_INLINE float max() const { return _mmVCL_hmax_ps(get(0)); }
	};
	
	VCL_STRONG_INLINE VectorScalar<float, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<float, 8>& a, const VectorScalar<float, 8>& b)
	{
		// Straight forward method
		// (b & ~mask) | (a & mask)
		//return VectorScalar<float, 8>(_mm256_or_ps(_mm256_andnot_ps(mask.get(0), b.get(0)), _mm256_and_ps(mask.get(0), a.get(0))));

		// xor-method
		// (((b ^ a) & mask)^b)
		//return VectorScalar<float, 8>(_mm256_xor_ps(b.get(0), _mm256_and_ps(mask.get(0), _mm256_xor_ps(b.get(0), a.get(0)))));

		// AVX way
		return VectorScalar<float, 8>(_mm256_blendv_ps(b.get(0), a.get(0), mask.get(0)));
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 8>& rhs)
	{
		alignas(32) float vars[8];
		_mm256_store_ps(vars, rhs.get(0));
		
		s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3]
				 << vars[4] << ", " << vars[5] << ", " << vars[6] << ", " << vars[7] << "'";

		return s;
	}
}
