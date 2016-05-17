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
#include <vector>

// VCL
#include <vcl/core/3rdparty/any.hpp>
#include <vcl/core/3rdparty/gsl/span.h>
#include <vcl/rtti/attribute.h>
#include <vcl/rtti/constructor.h>
#include <vcl/util/hashedstring.h>

namespace Vcl { namespace RTTI 
{
	// Based on the article series:
	// http://seanmiddleditch.com/journal/2012/01/c-metadata-part-i-singletons-and-lookup/
	class Type
	{
	public:
		Type(const char* name, size_t size, size_t alignment);
		Type(const char* name, size_t hash, size_t size, size_t alignment);
		Type(const Type&) = delete;
		Type(Type&&);
		virtual ~Type();

	public:
		Type& operator= (const Type&) = delete;

	public: // Properties
		const char* name() const { return _name; }
		size_t hash() const { return _hash; }

		size_t nrParents() const { return _parents.size(); }
		const Type* const* parents() const { return _parents.data(); }

		bool hasAttribute(const char* name) const;
		const AttributeBase* attribute(const char* name) const;

		/*!
		 * \brief Access the list of all attributes
		 */
		gsl::span<const std::unique_ptr<AttributeBase>> attributes() const { return _attributes; }

		const ConstructorSet& constructors() const { return _constructors; }

	public: // Queries
		bool isA(const Type* base) const;

	public: // Serialization
		void serialize(Serializer& ser, const void* obj) const;

	public:
		/// Allocate memory for a new instance of this type
		void* allocate() const;

		/// Free the memory of a desctructed object
		void deallocate(void* ptr) const;

		/// Call the constructor for a new instance of this type
		template<typename... Args>
		void construct(void* ptr, Args... args) const
		{
			_constructors.call(ptr, args...);
		}

		/// Destruct an instance of this type
		virtual void destruct(void* ptr) const;
		
	private:
		//! Readable type name
		const char* _name;

		//! Hash of the type name
		size_t _hash;

		//! Size of an instance
		size_t _size;

		//! Required alignment for an instance
		size_t _alignment;

		//! Version number
		int _version;
		
	protected: // List of parent types
		std::vector<const Type*> _parents;

	protected: // List of constructors for this type
		ConstructorSet _constructors;

	protected: // List of type attributes
		std::vector<std::unique_ptr<AttributeBase>> _attributes;

	protected: // List of general methods
		std::vector<const void*> _methods;
	};
}}
