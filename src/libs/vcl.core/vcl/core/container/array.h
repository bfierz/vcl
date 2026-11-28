/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#include <tuple>
#include <utility>

namespace Vcl { namespace Core {
	namespace detail {
		template<typename T, typename... Args, size_t... Is>
		VCL_STRONG_INLINE constexpr auto make_array_from_tuple_helper(const std::tuple<Args...>& attributes, std::index_sequence<Is...>)
		{
			return std::array<T, sizeof...(Args)>{ (&(std::get<Is>(attributes)))... };
		}
	}

	template<typename T, typename... Args>
	VCL_STRONG_INLINE constexpr auto make_array_from_tuple(const std::tuple<Args...>& attributes)
	{
		return detail::make_array_from_tuple_helper<T>(attributes, std::index_sequence_for<Args...>{});
	}
}}
