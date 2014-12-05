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
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>


#if defined VCL_VECTORIZE_SSE
#	include <vcl/core/simd/memory_sse.h>
#endif //VCL_VECTORIZE_SSE

#if defined VCL_VECTORIZE_AVX
#	include <vcl/core/simd/memory_avx.h>
#endif //VCL_VECTORIZE_AVX

namespace Vcl
{
	VCL_STRONG_INLINE void load(Eigen::Vector3f& value, const Eigen::Vector3f* base)
	{
		value = base[0];
	}

	VCL_STRONG_INLINE void load(Eigen::Vector3i& value, const Eigen::Vector3i* base)
	{
		value = base[0];
	}

	VCL_STRONG_INLINE void store(Eigen::Vector3f* base, const Eigen::Vector3f& loaded)
	{
		base[0] = loaded;
	}

	VCL_STRONG_INLINE void store(Eigen::Vector3i* base, const Eigen::Vector3i& loaded)
	{
		base[0] = loaded;
	}

	template<typename Scalar, int Width>
	VectorScalar<Scalar, Width> gather(Scalar const * base, VectorScalar<int, Width>& vindex, const int scale)
	{
		VectorScalar<Scalar, Width> res;
		for (int i = 0; i < Width; i++)
		{
			res[i] = *(base + vindex[i] * scale);
		}

		return res;
	}

	template<typename Scalar>
	Scalar gather(Scalar const * base, int vindex, const int scale)
	{
		return *(base + vindex * scale);
	}

	template<typename Scalar, int Width, int Rows, int Cols>
	Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> gather
	(
		const Eigen::Matrix<Scalar, Rows, Cols>* base,
		VectorScalar<int, Width>& vindex,
		const int scale
	)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> res;

		for (int i = 0; i < Width; i++)
		{
			for (int c = 0; c < Cols; c++)
			{
				for (int r = 0; r < Rows; r++)
				{
					res(r, c)[i] = base[vindex[i] * scale](r, c);
				}
			}
		}

		return res;
	}

	template<typename Scalar, int Width, int Rows, int Cols, int Stride>
	Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> gather
	(
		const Vcl::Core::InterleavedArray<Scalar, Rows, Cols, Stride>& base,
		VectorScalar<int, Width>& vindex,
		const int scale
	)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> res;
		for (int c = 0; c < Cols; c++)
		{
			for (int r = 0; r < Rows; r++)
			{
				VectorScalar<Scalar, Width> tmp;
				for (int i = 0; i < Width; i++)
				{
					tmp[i] = base.at<Scalar>(vindex[i] * scale)(r, c);
				}

				res(r, c) = tmp;
			}
		}

		return res;
	}

	template<typename Scalar, int Rows, int Cols, int Stride>
	Eigen::Matrix<Scalar, Rows, Cols> gather(const Vcl::Core::InterleavedArray<Scalar, Rows, Cols, Stride>& base, int vindex, const int scale)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		return base.at<Scalar>(vindex * scale);
	}




	template<typename Scalar, int Width>
	void scatter(VectorScalar<Scalar, Width> value, Scalar const * base, VectorScalar<int, Width>& vindex, const int scale)
	{
		for (int i = 0; i < Width; i++)
		{
			*(base + vindex[i] * scale) = value[i];
		}
	}

	template<typename Scalar>
	void scatter(Scalar value, Scalar const * base, int vindex, const int scale)
	{
		*(base + vindex * scale) = value;
	}


	template<typename Scalar, int Width, typename Storage>
	void scatter
	(
		const Eigen::Matrix<VectorScalar<Scalar, Width>, Storage::Rows, Storage::Cols>& value,
		Storage& base,
		VectorScalar<int, Width>& vindex,
		const int scale
	)
	{
		static_assert(Storage::Rows != Vcl::Core::DynamicStride && Storage::Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		for (int c = 0; c < Storage::Cols; c++)
		{
			for (int r = 0; r < Storage::Rows; r++)
			{
				for (int i = 0; i < Width; i++)
				{
					Scalar val = value(r, c)[i];
					base.at<Scalar>(vindex[i] * scale)(r, c) = val;
				}
			}
		}
	}
	
	template<typename Scalar, int Width, int Rows, int Cols>
	void scatter
	(
		Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> value, 
		Eigen::Matrix<Scalar, Rows, Cols>* base,
		VectorScalar<int, Width>& vindex,
		const int scale
	)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		for (int i = 0; i < Width; i++)
		{
			for (int c = 0; c < Cols; c++)
			{
				for (int r = 0; r < Rows; r++)
				{
					base[vindex[i] * scale](r, c) = value(r, c)[i];
				}
			}
		}
	}

	template<typename Scalar, int Rows, int Cols, int Stride>
	void scatter(const Eigen::Matrix<Scalar, Rows, Cols>& value, Vcl::Core::InterleavedArray<Scalar, Rows, Cols, Stride>& base, int vindex, const int scale)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		base.at<Scalar>(vindex * scale) = value;
	}
}
