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
#include <algorithm>
#include <vector>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Core
{
	/*!
	 *	\note The heap order is ignored until the maximum size is reached
	 */
	template<typename T>
	class bucket_adapter
	{
	public:
		bucket_adapter(std::vector<T>& cont)
		: _data(cont)
		{
		}

		/*!
		 *	\brief Add a new entry.
		 *
		 *	\note The heap order is ignored until the maximum size is reached
		 */
		void push(const T& v)
		{
			if (_data.size() < _data.capacity())
			{
				_data.emplace_back(v);
				if (_data.size() == _data.capacity())
				{
					std::make_heap(std::begin(_data), std::end(_data));
				}
			}
			else
			{
				// Remove the element with the least priority
				std::pop_heap(std::begin(_data), std::end(_data));
				_data.pop_back();

				// Add the new element
				_data.emplace_back(v);
				std::push_heap(std::begin(_data), std::end(_data));
			}
		}

		//! \returns the element with the highest priority
		const T& top()
		{
			VclRequire(_data.size() == _data.capacity(), "Set has reached the maximum limit.");

			return _data.front();
		}

		//! \returns the size of the set
		size_t size() const
		{
			return _data.size();
		}

		std::vector<T>& container() { return _data; }

	private:
		std::vector<T>& _data;
	};
}}
