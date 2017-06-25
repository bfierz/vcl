/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
 
// VCL library configuration
#include <vcl/config/global.h>

// C++ standard library
#include <array>
#include <utility>

namespace Vcl { namespace Util
{
	template <typename T>
	T max(T a, T b, T c)
	{
		T max_val = (a > b) ? a : b;
		max_val = (max_val > c) ? max_val : c;
		return max_val;
	}

	template <typename T>
	T min(T a, T b, T c)
	{
		T min_val = (a < b) ? a : b;
		min_val = (min_val < c) ? min_val : c;
		return min_val;
	}

	template<typename T>
	void sort(T& a, T& b)
	{
		T x = std::min(a, b);
		T y = std::max(a, b);

		a = x;
		b = y;
	}

	template<typename T>
	void sort(std::array<T, 2>& arr)
	{
		sort(arr[0], arr[1]);
	}

	template<typename T>
	void sort(T& a, T& b, T& c)
	{
		if (a <= b && a <= c)
		{
			sort(b, c);
		}
		else if (b <= a && b <= c)
		{
			T x = a;
			T y = c;

			sort(x, y);

			a = b;
			b = x;
			c = y;
		}
		else
		{
			T x = a;
			T y = b;

			sort(x, y);

			a = c;
			b = x;
			c = y;
		}
	}

	template<typename T>
	void sort(std::array<T, 3>& arr)
	{
		sort(arr[0], arr[1], arr[2]);
	}

	template<typename T>
	void sort(T& a, T& b, T& c, T& d)
	{
		if (a <= b && a <= c && a <= d)
		{
			sort(b, c, d);
		}
		else if (b <= a && b <= c && b <= d)
		{
			T x = a;
			T y = c;
			T z = d;

			sort(x, y, z);

			a = b;
			b = x;
			c = y;
			d = z;
		}
		else if (c <= a && c <= b && c <= d)
		{
			T x = a;
			T y = b;
			T z = d;

			sort(x, y, z);

			a = c;
			b = x;
			c = y;
			d = z;
		}
		else
		{
			T x = a;
			T y = b;
			T z = c;

			sort(x, y, z);

			a = d;
			b = x;
			c = y;
			d = z;
		}
	}

	template<typename T>
	void sort(std::array<T, 4>& arr)
	{
		sort(arr[0], arr[1], arr[2], arr[3]);
	}
}}
