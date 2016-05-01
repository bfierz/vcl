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
#include <vcl/rtti/metatypebase.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/rtti/metatyperegistry.h>

namespace Vcl { namespace RTTI 
{
	Type::Type(const char* name, size_t size, size_t alignment)
	: Type(name, Vcl::Util::StringHash(name).hash(), size, alignment)
	{
	}

	Type::Type(const char* name, size_t hash, size_t size, size_t alignment)
	: _name(name)
	, _hash(hash)
	, _size(size)
	, _alignment(alignment)
	, _isConstructable(false)
	{
		TypeRegistry::add(this);
	}
	
	Type::Type(Type&& rhs)
	{
		if (rhs.hash())
			TypeRegistry::remove(&rhs);
				
		_name = rhs._name;
		_hash = rhs._hash;
		_size = rhs._size;
		_alignment = rhs._alignment;
		_isConstructable = rhs._isConstructable;
		
		rhs._name = 0;
		rhs._hash = 0;
		rhs._size = 0;
		rhs._alignment = 0;
		rhs._isConstructable = false;

		_parents = std::move(rhs._parents);
		_attributes = std::move(rhs._attributes);
		_methods = std::move(rhs._methods);
		
		TypeRegistry::add(this);
	}

	Type::~Type()
	{
		if (hash())
			TypeRegistry::remove(this);

		// Free allocated resources
		for (AttributeBase*& attrib : _attributes)
		{
			VCL_SAFE_DELETE(attrib);
		};
	}
		
	void* Type::allocate() const
	{
		void* obj = _mm_malloc(_size, _alignment);

		return obj;
	}

	void Type::deallocate(void* ptr) const
	{
		_mm_free(ptr);
	}

	void Type::destruct(void* ptr) const
	{
	}

	void Type::construct(void* ptr, const std::initializer_list<linb::any>& params) const
	{
	}

	bool Type::isA(const Type* base) const
	{
		const auto* meta = this;
		while (meta != nullptr)
		{
			if (meta == base)
				return true; // found a match

			if (meta->nrParents() > 0)
				meta = meta->parents()[0];
			else
				meta = nullptr;
		}
		return false; // no match found
	}

	bool Type::hasAttribute(const char* name) const
	{
		size_t hash = Vcl::Util::StringHash(name).hash();

		auto attribIt = std::find_if(begin(_attributes), end(_attributes), [hash] (AttributeBase* attrib)
		{
			return attrib->hash() == hash;
		});

		if (attribIt != _attributes.end())
			return true;
		else if (nrParents() > 0)
			return parents()[0]->hasAttribute(name);
		else
			return false;
	}

	const AttributeBase* Type::attribute(const char* name) const
	{
		Require(hasAttribute(name), "Attribute exists.");

		size_t hash = Vcl::Util::StringHash(name).hash();

		auto attribIt = std::find_if(begin(_attributes), end(_attributes), [hash] (AttributeBase* attrib)
		{
			return attrib->hash() == hash;
		});
		
		if (attribIt != _attributes.end())
			return *attribIt;
		else if (nrParents() > 0)
			return parents()[0]->attribute(name);
		else
			return nullptr;
	}
}}
