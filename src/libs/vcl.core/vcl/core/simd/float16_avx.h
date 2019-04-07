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
#include <vcl/core/simd/bool16_avx.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_avx.h>

namespace Vcl
{
	template<>
	class alignas(32) VectorScalar<float, 16> : protected Core::Simd::VectorScalarBase<float, 16, Core::Simd::SimdExt::AVX>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(AVX)

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> operator+ (const VectorScalar<float, 16>& rhs) const { return VectorScalar<float, 16>(_mm256_add_ps(_data[0], rhs._data[0]), _mm256_add_ps(_data[1], rhs._data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> operator- (const VectorScalar<float, 16>& rhs) const { return VectorScalar<float, 16>(_mm256_sub_ps(_data[0], rhs._data[0]), _mm256_sub_ps(_data[1], rhs._data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> operator* (const VectorScalar<float, 16>& rhs) const { return VectorScalar<float, 16>(_mm256_mul_ps(_data[0], rhs._data[0]), _mm256_mul_ps(_data[1], rhs._data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> operator/ (const VectorScalar<float, 16>& rhs) const { return VectorScalar<float, 16>(_mm256_div_ps(_data[0], rhs._data[0]), _mm256_div_ps(_data[1], rhs._data[1])); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator += (const VectorScalar<float, 16>& rhs)
		{
			_data[0] = _mm256_add_ps(_data[0], rhs._data[0]);
			_data[1] = _mm256_add_ps(_data[1], rhs._data[1]);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator -= (const VectorScalar<float, 16>& rhs)
		{
			_data[0] = _mm256_sub_ps(_data[0], rhs._data[0]);
			_data[1] = _mm256_sub_ps(_data[1], rhs._data[1]);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator *= (const VectorScalar<float, 16>& rhs)
		{
			_data[0] = _mm256_mul_ps(_data[0], rhs._data[0]);
			_data[1] = _mm256_mul_ps(_data[1], rhs._data[1]);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator /= (const VectorScalar<float, 16>& rhs)
		{
			_data[0] = _mm256_div_ps(_data[0], rhs._data[0]);
			_data[1] = _mm256_div_ps(_data[1], rhs._data[1]);
			return *this;
		}
		
	public:
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator== (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmpeq_ps(_data[0], rhs._data[0]),
				_mm256_cmpeq_ps(_data[1], rhs._data[1])
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 16> operator!= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmpneq_ps(_data[0], rhs._data[0]),
				_mm256_cmpneq_ps(_data[1], rhs._data[1])
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 16> operator< (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmplt_ps(_data[0], rhs._data[0]),
				_mm256_cmplt_ps(_data[1], rhs._data[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator<= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmple_ps(_data[0], rhs._data[0]),
				_mm256_cmple_ps(_data[1], rhs._data[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator> (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmpgt_ps(_data[0], rhs._data[0]),
				_mm256_cmpgt_ps(_data[1], rhs._data[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator>= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmpge_ps(_data[0], rhs._data[0]),
				_mm256_cmpge_ps(_data[1], rhs._data[1])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> abs()   const { return VectorScalar<float, 16>(_mm256_abs_ps  (_data[0]), _mm256_abs_ps  (_data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sin()   const { return VectorScalar<float, 16>(_mm256_sin_ps  (_data[0]), _mm256_sin_ps  (_data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> cos()   const { return VectorScalar<float, 16>(_mm256_cos_ps  (_data[0]), _mm256_cos_ps  (_data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> exp()   const { return VectorScalar<float, 16>(_mm256_exp_ps  (_data[0]), _mm256_exp_ps  (_data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> log()   const { return VectorScalar<float, 16>(_mm256_log_ps  (_data[0]), _mm256_log_ps  (_data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sgn()   const { return VectorScalar<float, 16>(_mm256_sgn_ps  (_data[0]), _mm256_sgn_ps  (_data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sqrt()  const { return VectorScalar<float, 16>(_mm256_sqrt_ps (_data[0]), _mm256_sqrt_ps (_data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> rcp()   const { return VectorScalar<float, 16>(_mmVCL_rcp_ps  (_data[0]), _mmVCL_rcp_ps  (_data[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> rsqrt() const { return VectorScalar<float, 16>(_mmVCL_rsqrt_ps(_data[0]), _mmVCL_rsqrt_ps(_data[1])); }

		VCL_STRONG_INLINE VectorScalar<float, 16> acos() const { return VectorScalar<float, 16>(_mm256_acos_ps(_data[0]), _mm256_acos_ps(_data[1])); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> min(const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				_mm256_min_ps(_data[0], rhs._data[0]),
				_mm256_min_ps(_data[1], rhs._data[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<float, 16> max(const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				_mm256_max_ps(_data[0], rhs._data[0]),
				_mm256_max_ps(_data[1], rhs._data[1])
			);
		}

		VCL_STRONG_INLINE float dot(const VectorScalar<float, 16>& rhs) const
		{
			return
				_mmVCL_dp_ps(_data[0], rhs._data[0]) +
				_mmVCL_dp_ps(_data[1], rhs._data[1]);
		}

		VCL_STRONG_INLINE float min() const
		{
			return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(_mmVCL_hmin_ps(get(0))), _mm_set_ss(_mmVCL_hmin_ps(get(1)))));
		}
		VCL_STRONG_INLINE float max() const
		{
			return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(_mmVCL_hmax_ps(get(0))), _mm_set_ss(_mmVCL_hmax_ps(get(1)))));
		}
	};

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 16>& rhs)
	{
		alignas(32) float vars[16];
		_mm256_store_ps(vars + 0, rhs.get(0));
		_mm256_store_ps(vars + 8, rhs.get(1));
		
		s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3]
				 << vars[4] << ", " << vars[5] << ", " << vars[6] << ", " << vars[7] << "'";

		return s;
	}

	VCL_STRONG_INLINE VectorScalar<float, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<float, 16>& a, const VectorScalar<float, 16>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<float, 16>
		(
			_mm256_blendv_ps(b.get(0), a.get(0), mask.get(0)),
			_mm256_blendv_ps(b.get(1), a.get(1), mask.get(1))
		);
	}

	VCL_STRONG_INLINE VectorScalar<bool, 16> isinf(const VectorScalar<float, 16>& x)
	{
		return VectorScalar<bool, 16>
		(
			_mm256_isinf_ps(x.get(0)),
			_mm256_isinf_ps(x.get(1))
		);
	}
}
