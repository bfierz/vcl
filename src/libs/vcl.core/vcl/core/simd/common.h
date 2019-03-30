/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2019 Basil Fierz
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

// C++ Standard Library
//#include <array>

// VCL 
//#include <vcl/core/simd/bool4_sse.h>
//#include <vcl/core/simd/vectorscalar.h>
//#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl { namespace Core { namespace Simd
{
	//! Type of vectorization extension
	enum class SimdExt
	{
		None, //!< No vectorization
		SSE,  //!< Intel SSE (128 bits)
		AVX,  //!< Intel AVX (256 bits)
		NEON, //!< ARM NEON (128 bits)
	};
	
	template<typename, SimdExt>
	struct SimdRegister
	{
		//! Type of values in registers
		using Type = void;

		//! Number of values in a register
		static const int Width = 0;

		template<typename... U>
		Type set(U...)
		{

		}
	};

#ifdef VCL_VECTORIZE_SSE
	template<>
	struct SimdRegister<float, SimdExt::SSE>
	{
		using Scalar = float;
		using Type = __m128;
		static const int Width = 4;
		
		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return _mm_set1_ps(s0);
		}
		VCL_STRONG_INLINE static Type set(Scalar s0, Scalar s1, Scalar s2, Scalar s3)
		{
			return _mm_set_ps(s3, s2, s1, s0);
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
	};
	template<>
	struct SimdRegister<int, SimdExt::SSE>
	{
		using Scalar = int;
		using Type = __m128i;
		static const int Width = 4;
		
		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return _mm_set1_epi32(s0);
		}
		VCL_STRONG_INLINE static Type set(Scalar s0, Scalar s1, Scalar s2, Scalar s3)
		{
			return _mm_set_epi32(s3, s2, s1, s0);
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
	};
	template<>
	struct SimdRegister<bool, SimdExt::SSE>
	{
		using Scalar = bool;
		using Type = __m128;
		static const int Width = 4;
		
		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			const int m0 = s0 ? -1 : 0;
			return _mm_castsi128_ps(_mm_set1_epi32(m0));
		}
		VCL_STRONG_INLINE static Type set(Scalar s0, Scalar s1, Scalar s2, Scalar s3)
		{
			const int m0 = s0 ? -1 : 0;
			const int m1 = s1 ? -1 : 0;
			const int m2 = s2 ? -1 : 0;
			const int m3 = s3 ? -1 : 0;
			return _mm_castsi128_ps(_mm_set_epi32(m3, m2, m1, m0));
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
	};
#endif

	template<typename Scalar, int Width, SimdExt Type>
	struct VectorScalarBase
	{
		using RegType = typename SimdRegister<Scalar, Type>::Type;
		static const int RegValues = SimdRegister<Scalar, Type>::Width;
		static const int Regs = Width / RegValues;
		 
		RegType _data[Regs];
		
		VCL_STRONG_INLINE RegType get(int i) const
		{
			VclRequire(0 <= i && i < Regs, "Access is in range.");
			return _data[i];
		}

		VCL_STRONG_INLINE void set(Scalar s)
		{
			for (int i = 0; i < Regs; i++)
				_data[i] = SimdRegister<Scalar, Type>::set(s);
		}

		template<typename... T>
		VCL_STRONG_INLINE void set(T... vals)
		{
			setImpl(SimdWidthTag<RegValues>{}, 0, vals...);
		}

	private:
		template<int W>
		struct SimdWidthTag
		{};

		VCL_STRONG_INLINE void setImpl(SimdWidthTag<4>, int i, Scalar s0, Scalar s1, Scalar s2, Scalar s3)
		{
			_data[i] = SimdRegister<Scalar, Type>::set(s0, s1, s2, s3);
		}

		template<typename... T>
		VCL_STRONG_INLINE void setImpl(SimdWidthTag<4> tag, int i, Scalar s0, Scalar s1, Scalar s2, Scalar s3, T... vals)
		{
			_data[i] = SimdRegister<Scalar, Type>::set(s0, s1, s2, s3);
			setImpl(tag, i+1, vals...);
		}
		
		VCL_STRONG_INLINE void setImpl(SimdWidthTag<8>, int i, Scalar s0, Scalar s1, Scalar s2, Scalar s3, Scalar s4, Scalar s5, Scalar s6, Scalar s7)
		{
			_data[i] = SimdRegister<Scalar, Type>::set(s0, s1, s2, s3, s4, s5, s6, s7);
		}

		template<typename... T>
		VCL_STRONG_INLINE void setImpl(SimdWidthTag<8> tag, int i, Scalar s0, Scalar s1, Scalar s2, Scalar s3, Scalar s4, Scalar s5, Scalar s6, Scalar s7, T... vals)
		{
			_data[i] = SimdRegister<Scalar, Type>::set(s0, s1, s2, s3, s4, s5, s6, s7);
			setImpl(tag, i+1, vals...);
		}

		template<int W>
		VCL_STRONG_INLINE void setImpl(SimdWidthTag<W>, int i, RegType v0)
		{
			_data[i] = SimdRegister<Scalar, Type>::set(v0);
		}

		template<int W, typename... T>
		VCL_STRONG_INLINE void setImpl(SimdWidthTag<W> tag, int i, RegType v0, T... vals)
		{
			_data[i] = SimdRegister<Scalar, Type>::set(v0);
			setImpl(tag, i+1, vals...);
		}
	};
}}}
