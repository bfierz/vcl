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
#include <unordered_map>

// VCL
#include <vcl/rtti/metatype.h>

namespace Vcl { namespace RTTI 
{
	class TypeRegistry
	{
	private:
		typedef std::unordered_map<size_t, const Type*> TypeMap;

	public:
		/// Add a new meta type instance to the manager
		static void add(const Type* meta);

		/// Remove a meta type from the manager
		static void remove(const Type* meta);

		/// Find an instance of a meta type object by name
		static const Type* get(const gsl::cstring_span<> name);

	private:
		static TypeMap& instance();
	};
}}

inline const Vcl::RTTI::Type* vcl_meta_type_by_name(const gsl::cstring_span<> name)
{
	return Vcl::RTTI::TypeRegistry::get(name);
}
