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

// VCL
#include <vcl/core/any.h>
#include <vcl/core/contract.h>
#include <vcl/util/hashedstring.h>

namespace Vcl { namespace RTTI 
{
	// Forward declaration
	class Serializer;
	class Deserializer;

	class AttributeBase
	{
	public:
		template<size_t N>
		VCL_CONSTEXPR_CPP14 AttributeBase(const char (&name)[N])
		: _name(name, N - 1)
		, _hash(Vcl::Util::StringHash(name).hash())
		{
		}

	public:
		virtual void set(void* object, const std::any& param) const = 0;
		virtual void set(void* object, const std::string& param) const = 0;

		virtual void get(void* object, void* param, void* result) const = 0;
		virtual void get(void* object, const std::string& param, void* result) const = 0;

		virtual void serialize(Serializer& ser, const void* object) const = 0;
		virtual void deserialize(Deserializer& ser, void* object) const = 0;

	public:
		gsl::cstring_span<> name() const { return _name; }
		size_t hash() const { return _hash; }
		
	public:
		bool isReference() const
		{
			return (_flags & 0x00000001) != 0;
		}

		bool isShared() const
		{
			return (_flags & 0x00000002) != 0;
		}

		bool hasSetter() const
		{
			return (_flags & 0x00000004) != 0;
		}

		bool hasGetter() const
		{
			return (_flags & 0x00000008) != 0;
		}

	protected:
		void setIsReference()
		{
			_flags |= 0x00000001;
		}

		void setIsShared()
		{
			_flags |= 0x00000002;
		}

		void setHasSetter()
		{
			_flags |= 0x00000004;
		}

		void setHasGetter()
		{
			_flags |= 0x00000008;
		}

	private:
		//! Readable attribute name
		gsl::cstring_span<> _name;

		//! Attribute name hash
		size_t _hash;

	private:
		//! Flags describing the content of the attribute
		uint32_t _flags{ 0 };
	};
}}
