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

namespace Vcl { namespace Geometry
{
	template<typename Derived, typename T = unsigned int>
	class GenericId
	{
	public:
		using IdType = T;

	public:
		static Derived InvalidId() { return Derived(0xffffffff); }

	public:
		GenericId() = default;
		explicit GenericId(T id) : _id(id) {}
		GenericId(const GenericId<Derived, T>& other) = default;

	public:
		GenericId<Derived, T>& operator=(const GenericId<Derived, T>& other) = default;

	public:
		T id() const { return _id; }
		bool isValid() const { return _id != InvalidId().id(); }

	public:
		bool operator<(const GenericId<Derived, T>& other) const
		{
			return _id < other._id;
		}

		bool operator<=(const GenericId<Derived, T>& other) const
		{
			return _id <= other._id;
		}

		bool operator>(const GenericId<Derived, T>& other) const
		{
			return _id > other._id;
		}

		bool operator>=(const GenericId<Derived, T>& other) const
		{
			return _id >= other._id;
		}

		bool operator==(const GenericId<Derived, T>& other) const
		{
			return _id == other._id;
		}

		bool operator!=(const GenericId<Derived, T>& other) const
		{
			return _id != other._id;
		}

	protected:
		T _id{ InvalidId().id() };
	};
}}

// Instantiate a generic, typed ID
#define VCL_CREATEID(type_name, idx_type_name) class type_name : public Vcl::Geometry::GenericId<type_name, idx_type_name> { public: type_name(){} explicit type_name(idx_type_name id) : GenericId<type_name, idx_type_name>(id) {}}
