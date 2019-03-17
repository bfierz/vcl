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

// VCL
#include <vcl/core/contract.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Mathematics
{
	template<typename T, int F>
	class fixed
	{
		using ValueType = T;
		static const bool IsSigned = std::numeric_limits<ValueType>::is_signed;

		//! Number of bits use for the fractional part of the fixed-point number
		static const int FractionalBits = F;

		//! Number of bits use for the integer part of the fixed-point number
		static const int IntegerBits =
			8*sizeof(ValueType) - FractionalBits - (IsSigned ? 1 : 0);

	private:
		//! Conversion factor between floating point and fixed-point numbers
		static const T ConvFactor = 1 << FractionalBits;
		
		//! Check size consistency of the fixed-point number type
		static_assert(8*sizeof(ValueType) ==
			IntegerBits + FractionalBits + (IsSigned ? 1 : 0),
			"Integer and fractional parts are well-sized.");
		
		explicit fixed(T x)
		: _data(x)
		{
		}

	public:
		//! The default constructor leaves the number undefined
		fixed()
		{
		}

		explicit fixed(float x)
		: _data(static_cast<ValueType>(x * ConvFactor))
		{
			VclRequire(equal(x / static_cast<float>(*this), 1.0f, 1e-3f),
				"Conversion to fixed-point is correct");
		}

		explicit fixed(double x)
		: _data(static_cast<ValueType>(x * ConvFactor))
		{
			VclRequire(equal(x / static_cast<double>(*this), 1.0f, 1e-3f),
				"Conversion to fixed-point is correct");
		}

		explicit operator float() const
		{
			return static_cast<float>(_data) / ConvFactor;
		}
		
		explicit operator double() const
		{
			return static_cast<double>(_data) / ConvFactor;
		}

		template<typename U>
		explicit operator U() const
		{
			static_assert(std::is_integral<U>::value, "Target type must be integer");
			return _data >> FractionalBits;
		}

		T data() const { return _data; }

		fixed operator-()
		{
			const T neg = -_data;
			return fixed(neg);
		}

		fixed& operator+=(const fixed& x)
		{
			_data += x._data;
			return *this;
		}
		fixed& operator-=(const fixed& x)
		{
			_data -= x._data;
			return *this;
		}
		fixed& operator*=(const fixed& x)
		{
			_data *= x._data;
			_data >>= FractionalBits;
			return *this;
		}
		fixed& operator/=(const fixed& x)
		{
			_data /= x._data;
			_data *= ConvFactor;
			return *this;
		}

		fixed operator+(const fixed& x) const
		{
			const T sum = _data + x._data;
			return fixed{sum};
		}
		fixed operator-(const fixed& x) const
		{
			const T diff = _data - x._data;
			return fixed{diff};
		}
		
		bool operator==(const fixed& x) const
		{
			return _data == x._data;
		}
		bool operator!=(const fixed& x) const
		{
			return _data != x._data;
		}

	private:
		T _data;
	};
}}
