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

// GSL
#include <gsl/string_span>

namespace Vcl { namespace RTTI
{
	class Serializer
	{
	public:
		virtual void beginType(const gsl::cstring_span<> name, int version) = 0;

		//! Denote that the current type is finished
		virtual void endType() = 0;

		virtual void writeAttribute(const gsl::cstring_span<>, const gsl::cstring_span<> value) = 0;
	};

	class Deserializer
	{
	public:
		virtual void beginType(const gsl::cstring_span<> name) = 0;

		//! Denote that the current type is finished
		virtual void endType() = 0;

		//! \returns the type string of the current object
		virtual std::string readType() = 0;

		//! \returns true if the current object has the queried attribute
		virtual bool hasAttribute(const gsl::cstring_span<> name) = 0;


		virtual std::string readAttribute(const gsl::cstring_span<> name) = 0;
	};
}}
