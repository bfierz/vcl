/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2019 Basil Fierz
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

// Abseil
#include <absl/types/span.h>

namespace stdext
{
	template<typename T>
	using span = absl::Span<T>;

	template <typename ElementType>
	span<ElementType> make_span(span<ElementType> s) noexcept
	{
		return s;
	}

	template <typename T, std::size_t N>
	span<T> make_span(T (&arr)[N]) noexcept
	{
		return span<T>{arr};
	}

	template <typename T, std::size_t N>
	span<T> make_span(std::array<T, N>& arr) noexcept
	{
		return stdext::span<T>{arr};
	}

	template <typename T, std::size_t N>
	span<const T> make_span(const std::array<T, N>& arr) noexcept
	{
		return span<const T>{arr};
	}

	template <typename T>
	span<T> make_span(T* arr, size_t N) noexcept
	{
		return span<T>{arr, N};
	}

	template <typename Container>
	span<typename Container::value_type> make_span(Container& cont)
	{
		return span<typename Container::value_type>{cont};
	}

	template <typename Container>
	span<const typename Container::value_type> make_span(const Container& cont)
	{
		return span<const typename Container::value_type>{cont};
	}
}

