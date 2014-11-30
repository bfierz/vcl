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
#include <vcl/core/contract.h>

namespace Vcl
{
	template<typename Scalar, int Width>
	class VectorScalar
	{
	public:
		VectorScalar() = default;
		explicit VectorScalar(Scalar s)
		{
			for (size_t i = 0; i < Width; i++)
				mData[i] = s;
		}

	public:
		Scalar& operator[] (int idx)
		{
			Require(0 <= idx && idx < Width, "Access is in range.");

			return mData[idx];
		}

		Scalar operator[] (int idx) const
		{
			Require(0 <= idx && idx < Width, "Access is in range.");

			return mData[idx];
		}

	public:
		VectorScalar<Scalar, Width> operator+ (const VectorScalar<Scalar, Width>& rhs) const
		{
			VectorScalar<Scalar, Width> res;

			for (size_t i = 0; i < Width; i++)
				res[i] = mData[i] + rhs[i];

			return res;
		}
		VectorScalar<Scalar, Width> operator- (const VectorScalar<Scalar, Width>& rhs) const
		{
			VectorScalar<Scalar, Width> res;

			for (size_t i = 0; i < Width; i++)
				res[i] = mData[i] - rhs[i];

			return res;
		}
		VectorScalar<Scalar, Width> operator* (const VectorScalar<Scalar, Width>& rhs) const
		{
			VectorScalar<Scalar, Width> res;

			for (size_t i = 0; i < Width; i++)
				res[i] = mData[i] * rhs[i];

			return res;
		}
		VectorScalar<Scalar, Width> operator/ (const VectorScalar<Scalar, Width>& rhs) const
		{
			VectorScalar<Scalar, Width> res;

			for (size_t i = 0; i < Width; i++)
				res[i] = mData[i] / rhs[i];

			return res;
		}

	private:
		Scalar mData[Width];
	};

	VCL_STRONG_INLINE float select(bool mask, float a, float b)
	{
		return mask ? a : b;
	}
}

#if defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)
#	include <vcl/core/simd/bool4_sse.h>
#	include <vcl/core/simd/float4_sse.h>
#endif

#if defined VCL_VECTORIZE_AVX
#	include <vcl/core/simd/bool8_avx.h>
#	include <vcl/core/simd/bool16_avx.h>
#	include <vcl/core/simd/float8_avx.h>
#	include <vcl/core/simd/float16_avx.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 32>
	{
	private:
		__m256 mF8[4];
	};
}
#elif defined VCL_VECTORIZE_SSE
#	include <vcl/core/simd/bool8_sse.h>
#	include <vcl/core/simd/bool16_sse.h>
#	include <vcl/core/simd/float8_sse.h>
#	include <vcl/core/simd/float16_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 32>
	{
	private:
		__m128 mF4[8];
	};
}
#endif
	
namespace Vcl
{
	VCL_STRONG_INLINE bool any(bool b)
	{
		return b;
	}

	VCL_STRONG_INLINE bool all(bool b)
	{
		return b;
	}

	template<typename SCALAR>
	VCL_STRONG_INLINE SCALAR select(bool mask, const SCALAR& a, const SCALAR& b)
	{
		return mask ? a : b;
	}

	template<typename SCALAR>
	VCL_STRONG_INLINE void cswap(bool mask, SCALAR& a, SCALAR& b)
	{
		SCALAR c = a;
		a = select(mask, b, a);
		b = select(mask, c, b);
	}
	template<typename SCALAR>
	VCL_STRONG_INLINE void cnswap(bool mask, SCALAR& a, SCALAR& b)
	{
		SCALAR c = -a;
		a = select(mask, b, a);
		b = select(mask, c, b);
	}

	template<typename Scalar, int Width>
	VCL_STRONG_INLINE void cswap(const VectorScalar<bool, Width>& mask, VectorScalar<Scalar, Width>& a, VectorScalar<Scalar, Width>& b)
	{
		VectorScalar<Scalar, Width> c = select(mask, b, a);
		VectorScalar<Scalar, Width> d = select(mask, a, b);
		a = c;
		b = d;
	}
	
	template<typename Scalar, int Width>
	VCL_STRONG_INLINE void cnswap(const VectorScalar<bool, Width>& mask, VectorScalar<Scalar, Width>& a, VectorScalar<Scalar, Width>& b)
	{
		VectorScalar<Scalar, Width> c = -a;
		a = select(mask, b, a);
		b = select(mask, c, b);
	}
	
	template<typename Scalar, int Width>
	inline void cswap(const VectorScalar<bool, Width>& mask, Eigen::Matrix<VectorScalar<Scalar, Width>, 3, 1>& a, Eigen::Matrix<VectorScalar<Scalar, Width>, 3, 1>& b)
	{
		cswap(mask, a(0), b(0));
		cswap(mask, a(1), b(1));
		cswap(mask, a(2), b(2));
	}

	template<typename Scalar, int Width>
	inline void cnswap(const VectorScalar<bool, Width>& mask, Eigen::Matrix<VectorScalar<Scalar, Width>, 3, 1>& a, Eigen::Matrix<VectorScalar<Scalar, Width>, 3, 1>& b)
	{
		cnswap(mask, a(0), b(0));
		cnswap(mask, a(1), b(1));
		cnswap(mask, a(2), b(2));
	}
	
	typedef VectorScalar<float,  4> float4;
	typedef VectorScalar<float,  8> float8;
	typedef VectorScalar<float, 16> float16;
	typedef VectorScalar<float, 32> float32;
}

namespace Eigen
{
	template<> struct NumTraits<Vcl::float4> : GenericNumTraits<Vcl::float4>
	{
		enum
		{
			IsInteger = std::numeric_limits<float>::is_integer,
			IsSigned = std::numeric_limits<float>::is_signed,
			IsComplex = 0,
			RequireInitialization = internal::is_arithmetic<float>::value ? 0 : 1,
			ReadCost = 1,
			AddCost = 1,
			MulCost = 1
		};

		EIGEN_STRONG_INLINE static float dummy_precision() { return 1e-5f; }
	};
	template<> struct NumTraits<Vcl::float8> : GenericNumTraits<Vcl::float8>
	{
		enum
		{
			IsInteger = std::numeric_limits<float>::is_integer,
			IsSigned = std::numeric_limits<float>::is_signed,
			IsComplex = 0,
			RequireInitialization = internal::is_arithmetic<float>::value ? 0 : 1,
			ReadCost = 1,
			AddCost = 1,
			MulCost = 1
		};

		EIGEN_STRONG_INLINE static float dummy_precision() { return 1e-5f; }
	};
	template<> struct NumTraits<Vcl::float16> : GenericNumTraits<Vcl::float16>
	{
		enum
		{
			IsInteger = std::numeric_limits<float>::is_integer,
			IsSigned = std::numeric_limits<float>::is_signed,
			IsComplex = 0,
			RequireInitialization = internal::is_arithmetic<float>::value ? 0 : 1,
			ReadCost = 1,
			AddCost = 1,
			MulCost = 1
		};

		EIGEN_STRONG_INLINE static float dummy_precision() { return 1e-5f; }
	};
	template<> struct NumTraits<Vcl::float32> : GenericNumTraits<Vcl::float32>
	{
		enum
		{
			IsInteger = std::numeric_limits<float>::is_integer,
			IsSigned = std::numeric_limits<float>::is_signed,
			IsComplex = 0,
			RequireInitialization = internal::is_arithmetic<float>::value ? 0 : 1,
			ReadCost = 1,
			AddCost = 1,
			MulCost = 1
		};

		EIGEN_STRONG_INLINE static float dummy_precision() { return 1e-5f; }
	};
}

template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> abs (const Vcl::VectorScalar<Scalar, Width>& x) { return x.abs(); }

namespace Eigen { namespace numext
{
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE const Vcl::VectorScalar<Scalar, Width>& conj(const Vcl::VectorScalar<Scalar, Width>& x)  { return x; }
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE const Vcl::VectorScalar<Scalar, Width>& real(const Vcl::VectorScalar<Scalar, Width>& x)  { return x; }
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> imag(const Vcl::VectorScalar<Scalar, Width>&)   { return Vcl::VectorScalar<Scalar, Width>(0); }	
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> abs2(const Vcl::VectorScalar<Scalar, Width>& x) { return x*x; }
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> sqrt(const Vcl::VectorScalar<Scalar, Width>& x) { return x.sqrt(); }
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> exp (const Vcl::VectorScalar<Scalar, Width>& x) { return x.exp(); }
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> log (const Vcl::VectorScalar<Scalar, Width>& x) { return x.log(); }
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> sin (const Vcl::VectorScalar<Scalar, Width>& x) { return x.sin(); }
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> cos (const Vcl::VectorScalar<Scalar, Width>& x) { return x.cos(); }
	template<typename Scalar, int Width> EIGEN_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> pow (const Vcl::VectorScalar<Scalar, Width>& x, const Vcl::VectorScalar<Scalar, Width>& y)  { return x.pow(y); }
}}
