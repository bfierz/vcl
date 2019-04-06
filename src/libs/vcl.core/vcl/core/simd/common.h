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

// C++ standard library
#include <type_traits>

// Abseil
#include <absl/utility/utility.h>

// VCL
#include <vcl/core/simd/intrinsics_sse.h>
#include <vcl/core/simd/intrinsics_avx.h>

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
		//! Scalar value type stored in registers
		using Scalar = void;

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
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			return _mmVCL_extract_ps(vec, i);
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
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			return _mmVCL_extract_epi32(vec, i);
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
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			return _mmVCL_extract_epi32(_mm_castps_si128(vec), i) != 0;
		}
	};
#endif
	
#ifdef VCL_VECTORIZE_AVX
	template<>
	struct SimdRegister<float, SimdExt::AVX>
	{
		using Scalar = float;
		using Type = __m256;
		static const int Width = 8;
		
		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return _mm256_set1_ps(s0);
		}
		VCL_STRONG_INLINE static Type set
		(
			Scalar s0, Scalar s1, Scalar s2, Scalar s3,
			Scalar s4, Scalar s5, Scalar s6, Scalar s7
		)
		{
			return _mm256_set_ps(s7, s6, s5, s4, s3, s2, s1, s0);
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			return _mmVCL_extract_ps(vec, i);
		}
	};
	template<>
	struct SimdRegister<int, SimdExt::AVX>
	{
		using Scalar = int;
		using Type = __m256i;
		static const int Width = 8;
		
		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return _mm256_set1_epi32(s0);
		}
		VCL_STRONG_INLINE static Type set
		(
			Scalar s0, Scalar s1, Scalar s2, Scalar s3,
			Scalar s4, Scalar s5, Scalar s6, Scalar s7
		)
		{
			return _mm256_set_epi32(s7, s6, s5, s4, s3, s2, s1, s0);
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			return _mmVCL_extract_epi32(vec, i);
		}
	};
	template<>
	struct SimdRegister<bool, SimdExt::AVX>
	{
		using Scalar = bool;
		using Type = __m256;
		static const int Width = 8;
		
		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			const int m0 = s0 ? -1 : 0;
			return _mm256_castsi256_ps(_mm256_set1_epi32(m0));
		}
		VCL_STRONG_INLINE static Type set
		(
			Scalar s0, Scalar s1, Scalar s2, Scalar s3,
			Scalar s4, Scalar s5, Scalar s6, Scalar s7
		)
		{
			const int m0 = s0 ? -1 : 0;
			const int m1 = s1 ? -1 : 0;
			const int m2 = s2 ? -1 : 0;
			const int m3 = s3 ? -1 : 0;
			const int m4 = s4 ? -1 : 0;
			const int m5 = s5 ? -1 : 0;
			const int m6 = s6 ? -1 : 0;
			const int m7 = s7 ? -1 : 0;
			return _mm256_castsi256_ps(_mm256_set_epi32(m7, m6, m5, m4, m3, m2, m1, m0));
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			return _mmVCL_extract_epi32(_mm256_castps_si256(vec), i) != 0;
		}
	};
#endif

	template<typename ScalarT, int Width, SimdExt Type>
	class VectorScalarBase
	{
	protected:
		using RegType = typename SimdRegister<ScalarT, Type>::Type;
		using Scalar = typename SimdRegister<ScalarT, Type>::Scalar;
		static constexpr int NrValues = Width;
		static constexpr int NrValuesPerReg = SimdRegister<Scalar, Type>::Width;
		static constexpr int NrRegs = NrValues / NrValuesPerReg;

		//! SIMD registers
		RegType _data[NrRegs];
		
		VCL_STRONG_INLINE RegType get(int i) const
		{
			VclRequire(0 <= i && i < NrRegs, "Access is in range.");
			return _data[i];
		}
		VCL_STRONG_INLINE Scalar operator[] (int idx) const
		{
			VclRequire(0 <= idx && idx < Width, "Access is in range.");
			return SimdRegister<Scalar, Type>::get(get(idx / NrValuesPerReg), idx % NrValuesPerReg);
		}

		VCL_STRONG_INLINE void set(Scalar s)
		{
			for (int i = 0; i < NrRegs; i++)
				_data[i] = SimdRegister<Scalar, Type>::set(s);
		}

		template<typename... T>
		VCL_STRONG_INLINE void set(T... vals)
		{
			setImpl(SimdWidthTag<NrValuesPerReg>{}, 0, vals...);
		}

		template<typename T, size_t N, size_t... Is>
		VCL_STRONG_INLINE void set(const T (&arr)[N], absl::index_sequence<Is...>)
		{
			set(arr[Is]...);
		}
		
		//! Assignment operator
		//! Define a custom assignment in order to support the compiler generating
		//! SIMD instructions for copying data
		VCL_STRONG_INLINE VectorScalarBase<Scalar, Width, Type>& operator= (const VectorScalarBase<Scalar, Width, Type>& rhs)
		{
			set(rhs._data, absl::make_index_sequence<NrRegs>{});
			return *this;
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

	namespace Details
	{
		//! Add two floats
		//! \param a First summand
		//! \param b Second summand
		//! \returns The sum of \a and \b
		//! Helper function to allow for code geneartion in the VectorScalar specializations
		VCL_STRONG_INLINE float add(float a, float b)
		{
			return a + b;
		}
	}
}}}

#define VCL_SIMD_CONSTRUCTORS() \
	VCL_STRONG_INLINE VectorScalar() = default; \
	VCL_STRONG_INLINE VectorScalar(Scalar s) { set(s); } \
	template<typename... S> explicit VCL_STRONG_INLINE VectorScalar(Scalar s0, Scalar s1, Scalar s2, Scalar s3, S... s) { \
		static_assert(absl::conjunction<std::is_convertible<S, Scalar>...>::value, "All parameters need to be convertible to the scalar base type"); \
		static_assert(sizeof...(S) == NrValues-4, "Wrong number number of parameters"); \
		set(s0, s1, s2, s3, static_cast<Scalar>(s)...); } \
	template<typename... V> VCL_STRONG_INLINE explicit VectorScalar(__m128 v0, V... v) { \
		static_assert(sizeof...(V) == NrRegs-1, "Wrong number number of parameters"); \
		set(v0, v...); }


#define VCL_SIMD_P1_1(op) op(get(0))
#define VCL_SIMD_P1_2(op) VCL_SIMD_P1_1(op), op(get(1))
#define VCL_SIMD_P1_4(op) VCL_SIMD_P1_2(op), op(get(2)), op(get(3))
#define VCL_SIMD_P1_8(op) VCL_SIMD_P1_4(op), op(get(4)), op(get(5)), op(get(6)), op(get(7))

#define VCL_SIMD_P2_1(op) op(get(0), rhs.get(0))
#define VCL_SIMD_P2_2(op) VCL_SIMD_P2_1(op), op(get(1), rhs.get(1))
#define VCL_SIMD_P2_4(op) VCL_SIMD_P2_2(op), op(get(2), rhs.get(2)), op(get(3), rhs.get(3))
#define VCL_SIMD_P2_8(op) VCL_SIMD_P2_4(op), op(get(4), rhs.get(4)), op(get(5), rhs.get(5)), op(get(6), rhs.get(6)), op(get(7), rhs.get(7))

#define VCL_SIMD_RED_P1_1(op1, op2, i) op1(get(i))
#define VCL_SIMD_RED_P1_2(op1, op2, i) op2(VCL_SIMD_RED_P1_1(op1, op2, i), VCL_SIMD_RED_P1_1(op1, op2, i+1))
#define VCL_SIMD_RED_P1_4(op1, op2, i) op2(VCL_SIMD_RED_P1_2(op1, op2, i), VCL_SIMD_RED_P1_2(op1, op2, i+2))
#define VCL_SIMD_RED_P1_8(op1, op2, i) op2(VCL_SIMD_RED_P1_4(op1, op2, i), VCL_SIMD_RED_P1_4(op1, op2, i+4))

#define VCL_SIMD_RED_P2_1(op1, op2, i) op1(get(i), rhs.get(i))
#define VCL_SIMD_RED_P2_2(op1, op2, i) op2(VCL_SIMD_RED_P2_1(op1, op2, i), VCL_SIMD_RED_P2_1(op1, op2, i+1))
#define VCL_SIMD_RED_P2_4(op1, op2, i) op2(VCL_SIMD_RED_P2_2(op1, op2, i), VCL_SIMD_RED_P2_2(op1, op2, i+2))
#define VCL_SIMD_RED_P2_8(op1, op2, i) op2(VCL_SIMD_RED_P2_4(op1, op2, i), VCL_SIMD_RED_P2_4(op1, op2, i+4))

#define VCL_SIMD_UNARY_OP(name, op, N) VCL_STRONG_INLINE Self name() const { return Self{VCL_PP_JOIN_2(VCL_SIMD_P1_, N)(op)}; }
#define VCL_SIMD_BINARY_OP(name, op, N) VCL_STRONG_INLINE Self name(const Self& rhs) const { return Self{VCL_PP_JOIN_2(VCL_SIMD_P2_, N)(op)}; }
#define VCL_SIMD_ASSIGN_OP(name, op, N) VCL_STRONG_INLINE Self& name(const Self& rhs) { set(VCL_PP_JOIN_2(VCL_SIMD_P2_, N)(op)); return *this; }
#define VCL_SIMD_COMP_OP(name, op, N) VCL_STRONG_INLINE Bool name(const Self& rhs) const { return Bool{VCL_PP_JOIN_2(VCL_SIMD_P2_, N)(op)}; }
#define VCL_SIMD_QUERY_OP(name, op, N) VCL_STRONG_INLINE Bool name() const { return Bool{VCL_PP_JOIN_2(VCL_SIMD_P1_, N)(op)}; }
#define VCL_SIMD_UNARY_REDUCTION_OP(name, op1, op2, N) VCL_STRONG_INLINE Scalar name() const { return Scalar{VCL_PP_JOIN_2(VCL_SIMD_RED_P1_, N)(op1, op2, 0)}; }
#define VCL_SIMD_BINARY_REDUCTION_OP(name, op1, op2, N) VCL_STRONG_INLINE Scalar name(const Self& rhs) const { return Scalar{VCL_PP_JOIN_2(VCL_SIMD_RED_P2_, N)(op1, op2, 0)}; }
