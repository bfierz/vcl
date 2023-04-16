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

// C++ standard library
#include <array>

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>

#if defined VCL_VECTORIZE_SSE
#	include <vcl/core/simd/memory_sse.h>
#endif //VCL_VECTORIZE_SSE

#if defined VCL_VECTORIZE_AVX
#	include <vcl/core/simd/memory_avx.h>
#endif //VCL_VECTORIZE_AVX

#if defined VCL_VECTORIZE_NEON
#	include <vcl/core/simd/memory_neon.h>
#endif //VCL_VECTORIZE_NEON

namespace Vcl {
	VCL_STRONG_INLINE void load(float& value, const float* base) noexcept
	{
		VclRequire(base, "Load memory location is not null");

		value = base[0];
	}

	VCL_STRONG_INLINE void load(Eigen::Vector2f& value, const Eigen::Vector2f* base)
	{
		value = base[0];
	}

	VCL_STRONG_INLINE void load(Eigen::Vector2i& value, const Eigen::Vector2i* base)
	{
		value = base[0];
	}

	VCL_STRONG_INLINE void load(Eigen::Vector3f& value, const Eigen::Vector3f* base)
	{
		value = base[0];
	}

	VCL_STRONG_INLINE void load(Eigen::Vector3i& value, const Eigen::Vector3i* base)
	{
		value = base[0];
	}

	VCL_STRONG_INLINE void load(Eigen::Vector4f& value, const Eigen::Vector4f* base)
	{
		value = base[0];
	}

	VCL_STRONG_INLINE void load(Eigen::Vector4i& value, const Eigen::Vector4i* base)
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

	template<typename Scalar>
	Scalar gather(Scalar const* base, int vindex) noexcept
	{
		return *(base + vindex * 1);
	}

	template<typename Scalar, int Width, int Rows, int Cols>
	Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> gather(
		const Eigen::Matrix<Scalar, Rows, Cols>* base,
		VectorScalar<int, Width>& vindex)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");
		static_assert(sizeof(Eigen::Matrix<Scalar, Rows, Cols>) == Rows * Cols * sizeof(Scalar), "Size of matrix type does not contain any padding.");

		using wideint_t = VectorScalar<int, Width>;

		Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> res;

		for (int c = 0; c < Cols; c++)
		{
			for (int r = 0; r < Rows; r++)
			{
				const wideint_t scale = wideint_t(Rows * Cols);
				const wideint_t offset = wideint_t(Rows * c + r);
				const wideint_t idx = scale * vindex + offset;
				res(r, c) = gather(base->data(), idx);
			}
		}

		return res;
	}

	template<typename Scalar, int Width, int Rows, int Cols, int Stride>
	Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols> gather(
		const Vcl::Core::InterleavedArray<Scalar, Rows, Cols, Stride>& base,
		VectorScalar<int, Width>& vindex)
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
					tmp[i] = base.template at<Scalar>(vindex[i] * 1)(r, c);
				}

				res(r, c) = tmp;
			}
		}

		return res;
	}

	template<typename Scalar, int Rows, int Cols, int Stride>
	Eigen::Matrix<Scalar, Rows, Cols> gather(const Vcl::Core::InterleavedArray<Scalar, Rows, Cols, Stride>& base, int vindex)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		return base.template at<Scalar>(vindex * 1);
	}

	template<typename Scalar>
	void scatter(Scalar value, Scalar* base, int vindex) noexcept
	{
		*(base + vindex * 1) = value;
	}

	template<typename Scalar, int Rows, int Cols>
	void scatter(Eigen::Matrix<Scalar, Rows, Cols> value, Eigen::Matrix<Scalar, Rows, Cols>* base, int vindex) noexcept
	{
		*(base + vindex * 1) = value;
	}

	template<typename Scalar, int Width>
	void scatter(const VectorScalar<Scalar, Width>& value, Scalar* base, const VectorScalar<int, Width>& vindex) noexcept
	{
		for (int i = 0; i < Width; i++)
		{
			*(base + vindex[i] * 1) = value[i];
		}
	}

	template<typename Scalar, int Width, int Rows, int Cols>
	void scatter(
		const Eigen::Matrix<VectorScalar<Scalar, Width>, Rows, Cols>& value,
		Eigen::Matrix<Scalar, Rows, Cols>* base,
		const VectorScalar<int, Width>& vindex)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		using intN_t = VectorScalar<int, Width>;
		for (int c = 0; c < Cols; c++)
		{
			for (int r = 0; r < Rows; r++)
			{
				const intN_t scale = intN_t(Rows * Cols);
				const intN_t offset = intN_t(Rows * c + r);
				const intN_t idx = scale * vindex + offset;
				scatter(value(r, c), base->data(), idx);
			}
		}
	}

	template<typename Scalar, int Rows, int Cols, int Stride>
	void scatter(const Eigen::Matrix<Scalar, Rows, Cols>& value, Vcl::Core::InterleavedArray<Scalar, Rows, Cols, Stride>& base, int vindex)
	{
		static_assert(Rows != Vcl::Core::DynamicStride && Cols != Vcl::Core::DynamicStride, "Only fixed size matrices are supported.");

		base.template at<Scalar>(vindex * 1) = value;
	}

#if !defined(VCL_VECTORIZE_SSE) && !defined(VCL_VECTORIZE_AVX) && !defined(VCL_VECTORIZE_NEON)
	template<typename T, int Width>
	VCL_STRONG_INLINE void load(VectorScalar<T, Width>& value, const T* base)
	{
		value = VectorScalar<T, Width>(base, 1);
	}

	template<typename T, int Width>
	VCL_STRONG_INLINE void load(
		Eigen::Matrix<VectorScalar<T, Width>, 3, 1>& loaded,
		const Eigen::Matrix<T, 3, 1>* base)
	{
		loaded(0) = VectorScalar<T, Width>(base->data() + 0, 3);
		loaded(1) = VectorScalar<T, Width>(base->data() + 1, 3);
		loaded(2) = VectorScalar<T, Width>(base->data() + 2, 3);
	}

	template<typename T, int Width>
	VCL_STRONG_INLINE void load(
		Eigen::Matrix<VectorScalar<T, Width>, 4, 1>& loaded,
		const Eigen::Matrix<T, 4, 1>* base)
	{
		loaded(0) = VectorScalar<T, Width>(base->data() + 0, 4);
		loaded(1) = VectorScalar<T, Width>(base->data() + 1, 4);
		loaded(2) = VectorScalar<T, Width>(base->data() + 2, 4);
		loaded(3) = VectorScalar<T, Width>(base->data() + 3, 4);
	}

	template<typename T, int Width>
	VectorScalar<T, Width> gather(T const* base, VectorScalar<int, Width> vindex)
	{
		VectorScalar<T, Width> gathered;
		for (size_t i = 0; i < Width; i++)
			gathered[i] = *(base + vindex[i]);

		return gathered;
	}

	template<typename T, int Width>
	VCL_STRONG_INLINE std::array<VectorScalar<T, Width>, 2> interleave(const VectorScalar<T, Width>& a, const VectorScalar<T, Width>& b)
	{
		static_assert(Width % 2 == 0, "Interleaving requires even number of entries");

		VectorScalar<T, Width> low, high;
		for (size_t i = 0; i < Width / 2; i++)
		{
			low[2 * i + 0] = a[i];
			low[2 * i + 1] = b[i];
		}
		for (size_t i = 0; i < Width / 2; i++)
		{
			high[2 * i + 0] = a[Width / 2 + i];
			high[2 * i + 1] = b[Width / 2 + i];
		}

		return { low, high };
	}
#endif
}
