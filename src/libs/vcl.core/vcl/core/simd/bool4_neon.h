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
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

namespace Vcl
{
	template<>
	class VectorScalar<bool, 4>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(bool s)
		{
			_data[0] = s ? vdupq_n_u32((uint32_t) -1) : vdupq_n_u32(0);
		}
		explicit VCL_STRONG_INLINE VectorScalar(uint32x4_t F4)
		{
			_data[0] = F4;
		}
		
	public:
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator&& (const VectorScalar<bool, 4>& rhs) { return VectorScalar<bool, 4>(vandq_u32(_data[0], rhs._data[0])); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator|| (const VectorScalar<bool, 4>& rhs) { return VectorScalar<bool, 4>(vorrq_u32 (_data[0], rhs._data[0])); }

		VCL_STRONG_INLINE VectorScalar<bool, 4>& operator&= (const VectorScalar<bool, 4>& rhs) { _data[0] = vandq_u32(_data[0], rhs._data[0]); return *this; }
		VCL_STRONG_INLINE VectorScalar<bool, 4>& operator|= (const VectorScalar<bool, 4>& rhs) { _data[0] = vorrq_u32(_data[0], rhs._data[0]);  return *this; }

	public:
		friend VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b);
		friend VectorScalar<int, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<int, 4>& a, const VectorScalar<int, 4>& b);
		friend bool any(const VectorScalar<bool, 4>& b);
		friend bool all(const VectorScalar<bool, 4>& b);
		friend bool none(const VectorScalar<bool, 4>& b);

	private:
		uint32x4_t _data[1];
	};

	VCL_STRONG_INLINE bool any(const VectorScalar<bool, 4>& b)
	{
		return vmovemaskq_f32(b._data[0]) != 0;
	}

	VCL_STRONG_INLINE bool all(const VectorScalar<bool, 4>& b)
	{
		return static_cast<unsigned int>(vmovemaskq_f32(b._data[0])) == 0xf;
	}

	VCL_STRONG_INLINE bool none(const VectorScalar<bool, 4>& b)
	{
		return static_cast<unsigned int>(vmovemaskq_f32(b._data[0])) == 0x0;
	}
}
