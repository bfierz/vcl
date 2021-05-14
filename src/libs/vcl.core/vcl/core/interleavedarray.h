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

// VCL config
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// VCL
#include <vcl/core/memory/allocator.h>
#include <vcl/core/contract.h>

namespace Vcl { namespace Core
{
	const int DynamicStride = -1;

	/*!
	 *	Compute the width of the vector given its internal type
	 */
	template<typename T1, typename T2>
	struct VectorWidth
	{
		static const int value = sizeof(T1) / sizeof(T2);
	};

	/*!
	 *	Storage class storing the given matrix objects in row-major order.
	 */
	template<typename SCALAR, int ROWS = 0, int COLS = 0, int STRIDE = 0>
	class InterleavedArray
	{
	public:
		static const int Cols = COLS;
		static const int Rows = ROWS;
		static const int Stride = ((STRIDE == 0) || (STRIDE == 1)) ? 1 : STRIDE;

	public:
		/*!
		 *
		 *	\param size   Number of entries in the storage.
		 *	\param rows   Number of rows in a data element.
		 *	\param cols   Number of cols in a data element.
		 *	\param stride Defines the number elements being adjacent in memory.
		 *				  0: All data entries of an element are consecutive in memory.
		 *				  n: n elementy are grouped together. Note: 1 has the same effect as 0.
		 *				  Dynamic: The stride is the same as the number of elements in the container.
		 */
		InterleavedArray(size_t size, int rows = ROWS, int cols = COLS, int stride = STRIDE)
		: mSize(size)
		, mRows(static_cast<size_t>(rows))
		, mCols(static_cast<size_t>(cols))
		, mStride(static_cast<size_t>(stride))
		{
			// Template configuration checks
			static_assert(COLS == DynamicStride || COLS > 0, "Width of a data member is either dynamic or fixed sized.");
			static_assert(ROWS == DynamicStride || ROWS > 0, "Height of a data member is either dynamic or fixed sized.");

			// Initialisation checks
			VclRequire(rows > 0, "Number of rows is positive.");
			VclRequire(cols > 0, "Number of cols is positive.");
			VclRequire(stride == DynamicStride || stride >= 0, "Stride is Dynamic, 0 or greater 0");

			// Pad the requested size to the alignment
			// Note: This is done in order to support optimaly sized vector operations
			mAllocated = mSize;

			// Add enough memory to compensate the stride size
			if (stride != DynamicStride && mStride > 1)
			{
				if (mAllocated % mStride > 0)
					mAllocated += mStride - mAllocated % mStride;
			}

			const size_t alignment = 64;
			if (mAllocated % alignment > 0)
				mAllocated += alignment - mAllocated % alignment;

			// Allocate initial memory
			AlignedAllocPolicy<SCALAR, 64> alloc;
			mData = alloc.allocate(mAllocated*mRows*mCols);
		}

		InterleavedArray(InterleavedArray&& rhs)
		: mData(nullptr)
		, mSize(0)
		, mAllocated(0)
		, mRows(1)
		, mCols(1)
		, mStride(1)
		{
			std::swap(mData, rhs.mData);
			std::swap(mSize, rhs.mSize);
			std::swap(mAllocated, rhs.mAllocated);
			std::swap(mRows, rhs.mRows);
			std::swap(mCols, rhs.mCols);
			std::swap(mStride, rhs.mStride);
		}

		~InterleavedArray()
		{
			if (mData)
			{
				AlignedAllocPolicy<SCALAR, 64> alloc;
				alloc.deallocate(mData, mAllocated*mRows*mCols);
			}
		}

	public:
		SCALAR* data() const
		{
			return mData;
		}

		size_t size() const
		{
			return mSize;
		}

		size_t capacity() const
		{
			return mAllocated;
		}

		void setZero()
		{
			memset(mData, 0, mAllocated*mRows*mCols*sizeof(SCALAR));
		}

	public:
		template<typename SCALAR_OUT>
		Eigen::Map
		<
			Eigen::Matrix<SCALAR_OUT, ROWS, COLS>,
			Eigen::Unaligned,
			Eigen::Stride
			<
				((STRIDE == DynamicStride) ? DynamicStride : ((STRIDE == 0 || STRIDE == 1) ? ROWS : (ROWS*STRIDE / VectorWidth<SCALAR_OUT, SCALAR>::value))),
				((STRIDE == DynamicStride) ? DynamicStride : ((STRIDE == 0 || STRIDE == 1) ? 1 : (STRIDE / VectorWidth<SCALAR_OUT, SCALAR>::value)))
			>
		> at(size_t idx)
		{
			static_assert
			(
				(sizeof(SCALAR_OUT) >= sizeof(SCALAR)) && (sizeof(SCALAR_OUT) % sizeof(SCALAR) == 0),
				"Always access multiples of the internal type."
			);
			static_assert
			(
				implies(Stride != DynamicStride, (sizeof(SCALAR_OUT) / sizeof(SCALAR) <= static_cast<size_t>(Stride)) && (static_cast<size_t>(Stride) % (sizeof(SCALAR_OUT) / sizeof(SCALAR)) == 0)),
				"Output size and stride size are compatible."
			);

			typedef Eigen::Stride
			<
				((STRIDE == DynamicStride) ? DynamicStride : ((STRIDE == 0 || STRIDE == 1) ? ROWS : (ROWS*STRIDE / VectorWidth<SCALAR_OUT, SCALAR>::value))),
				((STRIDE == DynamicStride) ? DynamicStride : ((STRIDE == 0 || STRIDE == 1) ? 1 : (STRIDE / VectorWidth<SCALAR_OUT, SCALAR>::value)))
			> StrideType;

			VclRequire
			(
				implies(Stride == DynamicStride, (sizeof(SCALAR_OUT) / sizeof(SCALAR) <= mAllocated) && (mAllocated % (sizeof(SCALAR_OUT) / sizeof(SCALAR)) == 0)),
				"Output size and stride size are compatible."
			);

			const size_t rows = mRows;
			const size_t cols = mCols;

			// Stride between to entries of the same matrix
			const size_t stride = (mStride == static_cast<size_t>(DynamicStride)) ? mAllocated : mStride;

			// Size of a single entry
			const size_t scalar_width = sizeof(SCALAR_OUT) / sizeof(SCALAR);

			// Outer/Inner stride
			size_t outer_stride = 0;
			size_t inner_stride = 0;

			// Compute the address of the first entry
			auto base = reinterpret_cast<SCALAR_OUT*>(mData);
			if (stride == 0 || stride == 1)
			{
				base += idx*rows*cols;
			}
			else if (stride < mAllocated)
			{
				size_t entry = idx % stride;
				size_t group_idx = idx - entry;
				base += group_idx*rows*cols + entry;

				outer_stride = rows*stride / scalar_width;
				inner_stride = stride / scalar_width;
			}
			else
			{
				base += idx;

				outer_stride = rows*stride / scalar_width;
				inner_stride = stride / scalar_width;
			}

			if (Stride == DynamicStride || mStride == size_t(DynamicStride))
				return Eigen::Map<Eigen::Matrix<SCALAR_OUT, ROWS, COLS>, Eigen::Unaligned, StrideType>
				(
					base, StrideType(static_cast<ptrdiff_t>(outer_stride), static_cast<ptrdiff_t>(inner_stride))
				);
			else
				return Eigen::Map<Eigen::Matrix<SCALAR_OUT, ROWS, COLS>, Eigen::Unaligned, StrideType>(base);
		}

		template<typename SCALAR_OUT>
		const Eigen::Map
		<
			Eigen::Matrix<SCALAR_OUT, ROWS, COLS>,
			Eigen::Unaligned,
			Eigen::Stride
			<
				((STRIDE == DynamicStride) ? DynamicStride : ((STRIDE == 0 || STRIDE == 1) ? ROWS : (ROWS*STRIDE / VectorWidth<SCALAR_OUT, SCALAR>::value))),
				((STRIDE == DynamicStride) ? DynamicStride : ((STRIDE == 0 || STRIDE == 1) ? 1 : (STRIDE / VectorWidth<SCALAR_OUT, SCALAR>::value)))
			>
		> at(size_t idx) const
		{
			return const_cast<InterleavedArray*>(this)->at<SCALAR_OUT>(idx);
		}

	private:
		//! Pointer to the allocated memory
		SCALAR* mData;
		//! Number of used elements in the buffer
		size_t mSize;
		//! Number of allocated elements
		size_t mAllocated;
		//! Number of rows within each elements
		size_t mRows;
		//! Number of columns within each element
		size_t mCols;
		//! Number of elements being consecutive in memory
		size_t mStride;
	};
}}
