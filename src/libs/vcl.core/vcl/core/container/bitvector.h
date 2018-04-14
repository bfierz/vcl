/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#include <limits>
#include <vector>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Core
{	
	/*!
	 *	Picking up an idea formulated in:
	 *	http://upcoder.com/9/fast-resettable-flag-vector/
	 */
	template<template<class> class AllocatorT = std::allocator>
	class BitVector
	{
	public:
		using allocator_t = AllocatorT<uint16_t>;
		using container_t = std::vector<uint16_t, allocator_t>;

	public:
		class reference
		{
			friend class BitVector;

			reference() = delete;
			reference(const reference&) = delete;

			reference(uint16_t* ptr, const uint16_t* generation)
			: _dataPtr(ptr)
			, _generationPtr(generation)
			{
				VclRequire(ptr, "Pointer is valid.");
				VclRequire(generation, "Pointer is valid.");
			}

		public:
			~reference() = default;

			operator bool() const noexcept
			{
				return *_dataPtr == *_generationPtr;
			}

			reference& operator= (const bool x) noexcept
			{
				*_dataPtr = x ? *_generationPtr : 0;

				return *this;
			}

			reference& operator= (const reference& x) noexcept
			{
				*_dataPtr = *x._dataPtr;
				_generationPtr = x._generationPtr;

				return *this;
			}

			void flip()
			{
				if (*_dataPtr == *_generationPtr)
				{
					*_dataPtr = 0;
				}
				else
				{
					*_dataPtr = *_generationPtr;
				}
			}

		private:
			uint16_t* _dataPtr;
			const uint16_t* _generationPtr;
		};

	public:
		explicit BitVector(const allocator_t& alloc = allocator_t())
		: _generation(1)
		, _bits(alloc)
		{

		}

		explicit BitVector(size_t n, const allocator_t& alloc = allocator_t())
		: _generation(1)
		, _bits(n, alloc)
		{
		}

		BitVector(size_t n, bool val, const allocator_t& alloc = allocator_t())
		: _generation(1)
		, _bits(n, val, alloc)
		{

		}

	public: // Element access
		reference operator[] (size_t idx)
		{
			VclRequire(idx < _bits.size(), "Index is valid");

			return{ _bits.data() + idx, &_generation };
		}

		const reference operator[] (size_t idx) const
		{
			VclRequire(idx < _bits.size(), "Index is valid");

			return{ const_cast<uint16_t*>(_bits.data()) + idx, &_generation };
		}

	public: // Modifiers
		void clear()
		{
			_bits.clear();
		}

		void assign(size_t n, bool val)
		{
			VclRequire(n > 0, "Size is greater than zero.");

			// If the size is the same, try to increase the generation
			if (n == _bits.size() && val == false)
			{
				if (_generation == std::numeric_limits<uint16_t>::max())
				{
					_bits.assign(n, 0);
					_generation = 0;
				}
				++_generation;
			}
			else
			{
				// Reset the generation and assign every entry
				_generation = 1;

				if (val)
					_bits.assign(n, 1);
				else
					_bits.assign(n, 0);
			}
		}

		void setBit(size_t idx, bool val)
		{
			VclRequire(idx < _bits.size(), "Index is valid");

			_bits[idx] = val ? _generation : 0;
		}

		void resetAllBitsToFalse()
		{
			assign(_bits.size(), false);
		}

	public:
		size_t size() const noexcept
		{
			return _bits.size();
		}

		void shrink_to_fit()
		{
			_bits.shrink_to_fit();
		}

		uint16_t generation() const
		{
			return _generation;
		}

	private:
		//! Generation indicating the 'true' value.
		uint16_t _generation;

		//! Bits. Each short represents a single bit.
		container_t _bits;
	};
}}
