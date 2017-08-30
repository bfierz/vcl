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

#include <array>
#include <tuple>
#include <type_traits>

#if (defined(VCL_COMPILER_CLANG) || defined(VCL_COMPILER_GNU)) && __has_include(<experimental/array>)
#	include <experimental/array>
namespace std
{
	using std::experimental::make_array;
}
#else
////////////////////////////////////////////////////////////////////////////////
// http://en.cppreference.com/w/cpp/experimental/make_array
namespace std
{
#if defined(VCL_COMPILER_CLANG)
	template<class T> struct negation : integral_constant<bool, !static_cast<bool>(T::value)>{};
#endif

	namespace details
	{
		template<class> struct is_ref_wrapper : std::false_type {};
		template<class T> struct is_ref_wrapper<std::reference_wrapper<T>> : std::true_type {};

		template<class T>
		using not_ref_wrapper = std::negation<is_ref_wrapper<std::decay_t<T>>>;

		template <class D, class...> struct return_type_helper { using type = D; };
		template <class... Types>
		struct return_type_helper<void, Types...> : std::common_type<Types...> {
#if !defined(VCL_COMPILER_CLANG)
			static_assert(std::conjunction_v<not_ref_wrapper<Types>...>,
				"Types cannot contain reference_wrappers when D is void");
#endif
		};

		template <class D, class... Types>
		using return_type = std::array<typename return_type_helper<D, Types...>::type,
			sizeof...(Types)>;
	}

	template<class D = void, class... Types>
	VCL_STRONG_INLINE VCL_CONSTEXPR_CPP11 details::return_type<D, Types...> make_array(Types&&... t)
	{
		return{ std::forward<Types>(t)... };
	}
}
////////////////////////////////////////////////////////////////////////////////
#endif
	
namespace Vcl { namespace Core
{
	namespace detail
	{
		template<typename T, typename... Args, size_t... Is>
		VCL_STRONG_INLINE VCL_CONSTEXPR_CPP11 auto make_array_from_tuple_helper(const std::tuple<Args...>& attributes, std::index_sequence<Is...>)
		{
			return std::array<T, sizeof...(Args)>{ (&(std::get<Is>(attributes)))... };
		}
	}

	template<typename T, typename... Args>
	VCL_STRONG_INLINE VCL_CONSTEXPR_CPP11 auto make_array_from_tuple(const std::tuple<Args...>& attributes)
	{
		return detail::make_array_from_tuple_helper<T>(attributes, std::index_sequence_for<Args...>{});
	}
}}
