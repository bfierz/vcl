/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace Vcl { namespace Core
{
	//! Implements a container enumerator yielding item and index
	//! The original idea can be found here: https://blog.therocode.net/2018/10/for-each-with-index
	template <typename container_type>
	struct enumerate_wrapper
	{
		using iterator_type  = typename std::conditional<std::is_const<container_type>::value, typename container_type::const_iterator, typename container_type::iterator>::type;
		using pointer_type   = typename std::conditional<std::is_const<container_type>::value, typename container_type::const_pointer, typename container_type::pointer>::type;
		using reference_type = typename std::conditional<std::is_const<container_type>::value, typename container_type::const_reference, typename container_type::reference>::type;

		VCL_CPP_CONSTEXPR_14 enumerate_wrapper(container_type& c)
		: container(c)
		{
		}

		struct enumerate_wrapper_iter
		{
			size_t index;
			iterator_type value;

			VCL_CPP_CONSTEXPR_14 bool operator!=(const enumerate_wrapper_iter& other) const
			{
				return value != other.value;
			}
			VCL_CPP_CONSTEXPR_14 enumerate_wrapper_iter& operator++()
			{
				++index;
				++value;
				return *this;
			}

			VCL_CPP_CONSTEXPR_14 std::pair<size_t, reference_type> operator*() {
				return std::pair<size_t, reference_type>{index, *value};
			}
		};

		VCL_CPP_CONSTEXPR_14 enumerate_wrapper_iter begin()
		{
			return {0, std::begin(container)};
		}

		VCL_CPP_CONSTEXPR_14 enumerate_wrapper_iter end()
		{
			return {std::numeric_limits<size_t>::max(), std::end(container)};
		}
		container_type& container;
	};

	template <typename container_type>
	VCL_CPP_CONSTEXPR_14 auto enumerate(container_type& c)
	{
		return enumerate_wrapper<container_type>(c);
	}

	template <typename container_type>
	VCL_CPP_CONSTEXPR_14 auto const_enumerate(const container_type& c)
	{
		return enumerate_wrapper<const container_type>(c);
	}
}}
