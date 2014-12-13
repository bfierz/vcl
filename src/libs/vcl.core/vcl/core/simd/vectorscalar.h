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
#include <initializer_list>

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
		explicit VectorScalar(Scalar s[Width])
		{
			for (size_t i = 0; i < Width; i++)
				mData[i] = s[i];
		}
		explicit VectorScalar(std::initializer_list<Scalar> list)
		{
			int i = 0;
			for (Scalar s : list)
			{
				mData[i] = s;
				i++;
			}
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
}

#if defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)
#	include <vcl/core/simd/bool4_sse.h>
#	include <vcl/core/simd/float4_sse.h>
#	include <vcl/core/simd/int4_sse.h>
#endif

#if defined VCL_VECTORIZE_AVX
#	include <vcl/core/simd/bool8_avx.h>
#	include <vcl/core/simd/bool16_avx.h>
#	include <vcl/core/simd/float8_avx.h>
#	include <vcl/core/simd/float16_avx.h>

#	if defined VCL_VECTORIZE_AVX2
#		include <vcl/core/simd/int8_avx.h>
#		include <vcl/core/simd/int16_avx.h>
#	endif // defined VCL_VECTORIZE_AVX2

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
#	include <vcl/core/simd/int8_sse.h>
#	include <vcl/core/simd/int16_sse.h>

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
	template<typename Scalar, int Width>
	VCL_STRONG_INLINE VectorScalar<Scalar, Width> operator +(Scalar a, const VectorScalar<Scalar, Width>& b)
	{
		return VectorScalar<Scalar, Width>(a) + b;
	}
	template<typename Scalar, int Width>
	VCL_STRONG_INLINE VectorScalar<Scalar, Width> operator -(Scalar a, const VectorScalar<Scalar, Width>& b)
	{
		return VectorScalar<Scalar, Width>(a) -b;
	}
	template<typename Scalar, int Width>
	VCL_STRONG_INLINE VectorScalar<Scalar, Width> operator *(Scalar a, const VectorScalar<Scalar, Width>& b)
	{
		return VectorScalar<Scalar, Width>(a) * b;
	}
	template<typename Scalar, int Width>
	VCL_STRONG_INLINE VectorScalar<Scalar, Width> operator /(Scalar a, const VectorScalar<Scalar, Width>& b)
	{
		return VectorScalar<Scalar, Width>(a) / b;
	}


	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> abs(const Vcl::VectorScalar<Scalar, Width>& x) { return x.abs(); }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> abs2(const Vcl::VectorScalar<Scalar, Width>& x) { return x*x; }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> sqrt(const Vcl::VectorScalar<Scalar, Width>& x) { return x.sqrt(); }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> exp(const Vcl::VectorScalar<Scalar, Width>& x) { return x.exp(); }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> log(const Vcl::VectorScalar<Scalar, Width>& x) { return x.log(); }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> sgn(const Vcl::VectorScalar<Scalar, Width>& x) { return x.sgn(); }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> rcp(const Vcl::VectorScalar<Scalar, Width>& x) { return x.rcp(); }

	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> sin(const Vcl::VectorScalar<Scalar, Width>& x) { return x.sin(); }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> cos(const Vcl::VectorScalar<Scalar, Width>& x) { return x.cos(); }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> acos(const Vcl::VectorScalar<Scalar, Width>& x) { return x.acos(); }

	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> pow(const Vcl::VectorScalar<Scalar, Width>& x, const Vcl::VectorScalar<Scalar, Width>& y)  { return exp(log(x) * y); }

	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> min(const Vcl::VectorScalar<Scalar, Width>& x, const Vcl::VectorScalar<Scalar, Width>& y)  { return x.min(y); }
	template<typename Scalar, int Width> VCL_STRONG_INLINE Vcl::VectorScalar<Scalar, Width> max(const Vcl::VectorScalar<Scalar, Width>& x, const Vcl::VectorScalar<Scalar, Width>& y)  { return x.max(y); }



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

	template<typename Scalar, int Width, int Rows, int Cols>
	VCL_STRONG_INLINE Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> select
	(
		const VectorScalar<bool, Width>& mask,
		const Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols>& a,
		const Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols>& b
	)
	{
		Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> selected;
		for (int c = 0; c < Cols; c++)
		{
			for (int r = 0; r < Rows; r++)
			{
				selected(r, c) = select(mask, a(r, c), b(r, c));
			}
		}

		return selected;
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

	typedef VectorScalar<int,  4> int4;
	typedef VectorScalar<int,  8> int8;
	typedef VectorScalar<int, 16> int16;
	typedef VectorScalar<int, 32> int32;
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
