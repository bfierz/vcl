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
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl
{
	template<>
	class VectorScalar<bool, 8>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(bool s)
		{
			mF4[0] = s ? vdupq_n_u32((uint32_t) -1) : vdupq_n_u32(0);
			mF4[1] = s ? vdupq_n_u32((uint32_t) -1) : vdupq_n_u32(0);
		}
		explicit VCL_STRONG_INLINE VectorScalar(uint32x4_t I4_0, uint32x4_t I4_1)
		{
			mF4[0] = I4_0;
			mF4[1] = I4_1;
		}
		
	public:
		VectorScalar<bool, 8> operator&& (const VectorScalar<bool, 8>& rhs)
		{
			return VectorScalar<bool, 8>(vandq_u32(mF4[0], rhs.mF4[0]), vandq_u32(mF4[1], rhs.mF4[1]));
		}
		VectorScalar<bool, 8> operator|| (const VectorScalar<bool, 8>& rhs)
		{
			return VectorScalar<bool, 8>(vorrq_u32(mF4[0], rhs.mF4[0]), vorrq_u32(mF4[1], rhs.mF4[1]));
		}

		VCL_STRONG_INLINE VectorScalar<bool, 8>& operator&= (const VectorScalar<bool, 8>& rhs)
		{
			mF4[0] = vandq_u32(mF4[0], rhs.mF4[0]);
			mF4[1] = vandq_u32(mF4[1], rhs.mF4[1]);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8>& operator|= (const VectorScalar<bool, 8>& rhs)
		{
			mF4[0] = vorrq_u32(mF4[0], rhs.mF4[0]);
			mF4[1] = vorrq_u32(mF4[1], rhs.mF4[1]);
			return *this;
		}

	public:
		friend VectorScalar<float, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<float, 8>& a, const VectorScalar<float, 8>& b);
		friend VectorScalar<int, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<int, 8>& a, const VectorScalar<int, 8>& b);
		friend bool any(const VectorScalar<bool, 8>& b);
		friend bool all(const VectorScalar<bool, 8>& b);
		friend bool none(const VectorScalar<bool, 8>& b);

	private:
		uint32x4_t mF4[2];
	};

	VCL_STRONG_INLINE bool any(const VectorScalar<bool, 8>& b)
	{
		int mask  = vmovemaskq_f32(b.mF4[1]) << 4;
		    mask |= vmovemaskq_f32(b.mF4[0]);

		return mask != 0;
	}

	VCL_STRONG_INLINE bool all(const VectorScalar<bool, 8>& b)
	{
		int mask  = vmovemaskq_f32(b.mF4[1]) << 4;
		    mask |= vmovemaskq_f32(b.mF4[0]);
			
		return static_cast<unsigned int>(mask) == 0xff;
	}

	VCL_STRONG_INLINE bool none(const VectorScalar<bool, 8>& b)
	{
		int mask  = vmovemaskq_f32(b.mF4[1]) << 4;
		    mask |= vmovemaskq_f32(b.mF4[0]);

		return static_cast<unsigned int>(mask) == 0x0;
	}
}
