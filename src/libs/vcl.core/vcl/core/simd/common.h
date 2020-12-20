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
	};

	template<>
	struct SimdRegister<float, SimdExt::None>
	{
		using Scalar = float;
		using Type = float;
		static const int Width = 1;

		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return s0;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int)
		{
			return vec;
		}
	};
	template<>
	struct SimdRegister<int, SimdExt::None>
	{
		using Scalar = int;
		using Type = int;
		static const int Width = 1;

		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return s0;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int)
		{
			return vec;
		}
	};
	template<>
	struct SimdRegister<bool, SimdExt::None>
	{
		using Scalar = bool;
		using Type = bool;
		static const int Width = 1;

		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return s0;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int)
		{
			return vec;
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

#ifdef VCL_VECTORIZE_NEON
	template<>
	struct SimdRegister<float, SimdExt::NEON>
	{
		using Scalar = float;
		using Type = float32x4_t;
		static const int Width = 4;

		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return vdupq_n_f32(s0);
		}
		VCL_STRONG_INLINE static Type set(Scalar s0, Scalar s1, Scalar s2, Scalar s3)
		{
			alignas(8) float data[4] = { s0, s1, s2, s3 };
			return vld1q_f32(data);
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			switch (i)
			{
			case 0:
				return vgetq_lane_f32(vec, 0);
			case 1:
				return vgetq_lane_f32(vec, 1);
			case 2:
				return vgetq_lane_f32(vec, 2);
			case 3:
				return vgetq_lane_f32(vec, 3);
			}
			return 0;
		}
	};
	template<>
	struct SimdRegister<int, SimdExt::NEON>
	{
		using Scalar = int;
		using Type = int32x4_t;
		static const int Width = 4;

		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			return vdupq_n_s32(s0);
		}
		VCL_STRONG_INLINE static Type set(Scalar s0, Scalar s1, Scalar s2, Scalar s3)
		{
			alignas(8) int data[4] = { s0, s1, s2, s3 };
			return vld1q_s32(data);
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			switch (i)
			{
			case 0:
				return vgetq_lane_s32(vec, 0);
			case 1:
				return vgetq_lane_s32(vec, 1);
			case 2:
				return vgetq_lane_s32(vec, 2);
			case 3:
				return vgetq_lane_s32(vec, 3);
			}
			return 0;
		}
	};
	template<>
	struct SimdRegister<bool, SimdExt::NEON>
	{
		using Scalar = bool;
		using Type = uint32x4_t;
		static const int Width = 4;

		VCL_STRONG_INLINE static Type set(Scalar s0)
		{
			const uint32_t m0 = s0 ? uint32_t(-1) : 0;
			return vdupq_n_u32(m0);
		}
		VCL_STRONG_INLINE static Type set(Scalar s0, Scalar s1, Scalar s2, Scalar s3)
		{
			const uint32_t m0 = s0 ? uint32_t(-1) : 0;
			const uint32_t m1 = s1 ? uint32_t(-1) : 0;
			const uint32_t m2 = s2 ? uint32_t(-1) : 0;
			const uint32_t m3 = s3 ? uint32_t(-1) : 0;

			alignas(8) uint32_t data[4] = { m0, m1, m2, m3 };
			return vld1q_u32(data);
		}
		VCL_STRONG_INLINE static Type set(Type vec)
		{
			return vec;
		}
		VCL_STRONG_INLINE static Scalar get(Type vec, int i)
		{
			switch (i)
			{
			case 0:
				return vgetq_lane_u32(vec, 0) != 0;
			case 1:
				return vgetq_lane_u32(vec, 1) != 0;
			case 2:
				return vgetq_lane_u32(vec, 2) != 0;
			case 3:
				return vgetq_lane_u32(vec, 3) != 0;
			}
			return false;
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
		VCL_STRONG_INLINE void setImpl(SimdWidthTag<W>, int i, const RegType& v0)
		{
			_data[i] = SimdRegister<Scalar, Type>::set(v0);
		}

		template<int W, typename... T>
		VCL_STRONG_INLINE void setImpl(SimdWidthTag<W> tag, int i, const RegType& v0, T... vals)
		{
			_data[i] = SimdRegister<Scalar, Type>::set(v0);
			setImpl(tag, i+1, vals...);
		}
	};

	namespace Details
	{
		template<typename T>
		VCL_STRONG_INLINE T nop(T a)
		{
			return a;
		}
		//! Add two floats
		//! \tparam T Scalar type used for operation
		//! \param a First summand
		//! \param b Second summand
		//! \returns The sum of \a and \b
		//! Helper function to allow for code generation in the VectorScalar specializations
		template<typename T>
		VCL_STRONG_INLINE T add(T a, T b)
		{
			return a + b;
		}
		template<typename T>
		VCL_STRONG_INLINE T sub(T a, T b)
		{
			return a - b;
		}
		template<typename T>
		VCL_STRONG_INLINE T mul(T a, T b)
		{
			return a * b;
		}
		template<typename T>
		VCL_STRONG_INLINE T div(T a, T b)
		{
			return a / b;
		}

		template<typename T>
		VCL_STRONG_INLINE bool cmpeq(T a, T b)
		{
			return a == b;
		}
		template<typename T>
		VCL_STRONG_INLINE bool cmpne(T a, T b)
		{
			return a != b;
		}
		template<typename T>
		VCL_STRONG_INLINE bool cmplt(T a, T b)
		{
			return a < b;
		}
		template<typename T>
		VCL_STRONG_INLINE bool cmple(T a, T b)
		{
			return a <= b;
		}
		template<typename T>
		VCL_STRONG_INLINE bool cmpgt(T a, T b)
		{
			return a > b;
		}
		template<typename T>
		VCL_STRONG_INLINE bool cmpge(T a, T b)
		{
			return a >= b;
		}

		//! Logical conjunction
		VCL_STRONG_INLINE bool conj(bool a, bool b)
		{
			return a && b;
		}
		//! Bitwise conjunction
		VCL_STRONG_INLINE int conj(int a, int b)
		{
			return a & b;
		}
		//! Logical disjunction
		VCL_STRONG_INLINE bool disj(bool a, bool b)
		{
			return a || b;
		}
		//! Bitwise disjunction
		VCL_STRONG_INLINE int disj(int a, int b)
		{
			return a | b;
		}
	}
}}}

#define VCL_SIMD_VECTORSCALAR_SETUP(SIMD_TYPE) \
	using Base = Core::Simd::VectorScalarBase<Scalar, NrValues, Core::Simd::SimdExt::SIMD_TYPE>; \
	using Self = VectorScalar<Scalar, NrValues>; \
	using Bool = VectorScalar<bool, NrValues>; \
	using Base::operator[]; \
	using Base::get; \
	VCL_STRONG_INLINE VectorScalar() = default; \
	VCL_STRONG_INLINE VectorScalar(Scalar s) { set(s); } \
	template<typename... S> explicit VCL_STRONG_INLINE VectorScalar(Scalar s0, Scalar s1, Scalar s2, Scalar s3, S... s) { \
		static_assert(absl::conjunction<std::is_convertible<S, Scalar>...>::value, "All parameters need to be convertible to the scalar base type"); \
		static_assert(sizeof...(S) == NrValues-4, "Wrong number number of parameters"); \
		set(s0, s1, s2, s3, static_cast<Scalar>(s)...); } \
	template<typename... V> VCL_STRONG_INLINE explicit VectorScalar(const RegType& v0, V... v) { \
		static_assert(sizeof...(V) == NrRegs-1, "Wrong number number of parameters"); \
		set(v0, v...); }

#define VCL_SIMD_P1_1(op, i)  op(get(i))
#define VCL_SIMD_P1_2(op, i)  VCL_SIMD_P1_1(op, i), VCL_SIMD_P1_1(op, i+1)
#define VCL_SIMD_P1_4(op, i)  VCL_SIMD_P1_2(op, i), VCL_SIMD_P1_2(op, i+2)
#define VCL_SIMD_P1_8(op, i)  VCL_SIMD_P1_4(op, i), VCL_SIMD_P1_4(op, i+4)
#define VCL_SIMD_P1_16(op, i) VCL_SIMD_P1_8(op, i), VCL_SIMD_P1_8(op, i+8)

#define VCL_SIMD_P2_1(op, i)  op(get(i), rhs.get(i))
#define VCL_SIMD_P2_2(op, i)  VCL_SIMD_P2_1(op, i), VCL_SIMD_P2_1(op, i+1)
#define VCL_SIMD_P2_4(op, i)  VCL_SIMD_P2_2(op, i), VCL_SIMD_P2_2(op, i+2)
#define VCL_SIMD_P2_8(op, i)  VCL_SIMD_P2_4(op, i), VCL_SIMD_P2_4(op, i+4)
#define VCL_SIMD_P2_16(op, i) VCL_SIMD_P2_8(op, i), VCL_SIMD_P2_8(op, i+8)

#define VCL_SIMD_RED_P1_1(op1, op2, i)  op1(get(i))
#define VCL_SIMD_RED_P1_2(op1, op2, i)  op2(VCL_SIMD_RED_P1_1(op1, op2, i), VCL_SIMD_RED_P1_1(op1, op2, i+1))
#define VCL_SIMD_RED_P1_4(op1, op2, i)  op2(VCL_SIMD_RED_P1_2(op1, op2, i), VCL_SIMD_RED_P1_2(op1, op2, i+2))
#define VCL_SIMD_RED_P1_8(op1, op2, i)  op2(VCL_SIMD_RED_P1_4(op1, op2, i), VCL_SIMD_RED_P1_4(op1, op2, i+4))
#define VCL_SIMD_RED_P1_16(op1, op2, i) op2(VCL_SIMD_RED_P1_8(op1, op2, i), VCL_SIMD_RED_P1_8(op1, op2, i+8))

#define VCL_SIMD_RED_P2_1(op1, op2, i)  op1(get(i), rhs.get(i))
#define VCL_SIMD_RED_P2_2(op1, op2, i)  op2(VCL_SIMD_RED_P2_1(op1, op2, i), VCL_SIMD_RED_P2_1(op1, op2, i+1))
#define VCL_SIMD_RED_P2_4(op1, op2, i)  op2(VCL_SIMD_RED_P2_2(op1, op2, i), VCL_SIMD_RED_P2_2(op1, op2, i+2))
#define VCL_SIMD_RED_P2_8(op1, op2, i)  op2(VCL_SIMD_RED_P2_4(op1, op2, i), VCL_SIMD_RED_P2_4(op1, op2, i+4))
#define VCL_SIMD_RED_P2_16(op1, op2, i) op2(VCL_SIMD_RED_P2_8(op1, op2, i), VCL_SIMD_RED_P2_8(op1, op2, i+8))

#define VCL_SIMD_UNARY_OP(name, op, N) VCL_STRONG_INLINE Self name() const { return Self{VCL_PP_JOIN_2(VCL_SIMD_P1_, N)(op, 0)}; }
#define VCL_SIMD_BINARY_OP(name, op, N) VCL_STRONG_INLINE Self name(const Self& rhs) const { return Self{VCL_PP_JOIN_2(VCL_SIMD_P2_, N)(op, 0)}; }
#define VCL_SIMD_ASSIGN_OP(name, op, N) VCL_STRONG_INLINE Self& name(const Self& rhs) { set(VCL_PP_JOIN_2(VCL_SIMD_P2_, N)(op, 0)); return *this; }
#define VCL_SIMD_COMP_OP(name, op, N) VCL_STRONG_INLINE Bool name(const Self& rhs) const { return Bool{VCL_PP_JOIN_2(VCL_SIMD_P2_, N)(op, 0)}; }
#define VCL_SIMD_QUERY_OP(name, op, N) VCL_STRONG_INLINE Bool name() const { return Bool{VCL_PP_JOIN_2(VCL_SIMD_P1_, N)(op, 0)}; }
#define VCL_SIMD_UNARY_REDUCTION_OP(name, op1, op2, N) VCL_STRONG_INLINE Scalar name() const { return Scalar{VCL_PP_JOIN_2(VCL_SIMD_RED_P1_, N)(op1, op2, 0)}; }
#define VCL_SIMD_BINARY_REDUCTION_OP(name, op1, op2, N) VCL_STRONG_INLINE Scalar name(const Self& rhs) const { return Scalar{VCL_PP_JOIN_2(VCL_SIMD_RED_P2_, N)(op1, op2, 0)}; }
