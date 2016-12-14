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
#include <vcl/rtti/attributebase.h>
#include <vcl/rtti/metatyperegistry.h>
#include <vcl/rtti/serializer.h>
#include <vcl/util/hashedstring.h>

namespace Vcl { namespace RTTI 
{	
	Type::Type(Type&& rhs)
	: _name{ rhs._name }
	{
		if (rhs.hash())
			TypeRegistry::remove(&rhs);
				
		_hash = rhs._hash;
		_size = rhs._size;
		_alignment = rhs._alignment;
		_version = rhs._version;
		
		rhs._hash = 0;
		rhs._size = 0;
		rhs._alignment = 0;
		rhs._version = 1;

		_parents = std::move(rhs._parents);
		_constructors = std::move(rhs._constructors);
		_attributes = std::move(rhs._attributes);
		_methods = std::move(rhs._methods);
		
		TypeRegistry::add(this);
	}

	Type::~Type()
	{
		if (hash())
			TypeRegistry::remove(this);
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
		VCL_UNREFERENCED_PARAMETER(ptr);
	}

	bool Type::isA(const Type* base) const
	{
		const auto* meta = this;
		while (meta != nullptr)
		{
			if (meta->hash() == base->hash())
				return true; // found a match

			if (meta->nrParents() > 0)
				meta = meta->parents()[0];
			else
				meta = nullptr;
		}
		return false; // no match found
	}

	bool Type::hasAttribute(const gsl::cstring_span<> name) const
	{
		size_t hash = Vcl::Util::StringHash(name).hash();

		auto attribIt = std::find_if(begin(_attributes), end(_attributes), [hash] (const std::unique_ptr<AttributeBase>& attrib)
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

	const AttributeBase* Type::attribute(const gsl::cstring_span<> name) const
	{
		Require(hasAttribute(name), "Attribute exists.");

		size_t hash = Vcl::Util::StringHash(name).hash();

		auto attribIt = std::find_if(begin(_attributes), end(_attributes), [hash] (const std::unique_ptr<AttributeBase>& attrib)
		{
			return attrib->hash() == hash;
		});
		
		if (attribIt != _attributes.end())
			return attribIt->get();
		else if (nrParents() > 0)
			return parents()[0]->attribute(name);
		else
			return nullptr;
	}

	void Type::serialize(Serializer& ser, const void* obj) const
	{
		// Write out the type specific data
		// * Type name
		// * Version
		ser.beginType(_name, _version);

		// Serialize each attribute
		serializeAttributes(ser, obj);

		// Done
		ser.endType();
	}

	void Type::serializeAttributes(Serializer& ser, const void* obj) const
	{
		// Check the parents attributes
		for (const auto* p : _parents)
		{
			p->serializeAttributes(ser, obj);
		}

		// Serialize each attribute
		for (const auto& attr : _attributes)
		{
			attr->serialize(ser, obj);
		}
	}

	void Type::deserialize(Deserializer& deser, void* obj) const
	{
		// Check the parents attributes
		for (const auto* p : _parents)
		{
			p->deserialize(deser, obj);
		}

		// Deserialize each attribute
		for (const auto& attr : _attributes)
		{
			if (deser.hasAttribute(attr->name()))
			{
				attr->deserialize(deser, obj);
			}
		}
	}
}}
