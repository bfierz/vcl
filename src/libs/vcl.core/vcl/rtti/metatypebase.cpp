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
#include <vcl/core/memory/allocator.h>
#include <vcl/core/contract.h>
#include <vcl/rtti/attributebase.h>
#include <vcl/rtti/metatyperegistry.h>
#include <vcl/rtti/serializer.h>
#include <vcl/util/hashedstring.h>

namespace Vcl { namespace RTTI {
	Type::Type(stdext::string_view name, size_t hash, size_t size, size_t alignment)
	: _name(name)
	, _hash(hash)
	, _size(size)
	, _alignment(alignment)
	, _version(1)
	{
		TypeRegistry::add(this);
	}

	Type::Type(Type&& rhs) noexcept
	: _name{ rhs._name }
	{
		if (rhs.hash() != 0u)
		{
			TypeRegistry::remove(&rhs);
		}

		_hash = rhs._hash;
		_size = rhs._size;
		_alignment = rhs._alignment;
		_version = rhs._version;

		rhs._hash = 0;
		rhs._size = 0;
		rhs._alignment = 0;
		rhs._version = 1;

		std::swap(_parents, rhs._parents);
		std::swap(_constructors, rhs._constructors);
		std::swap(_attributes, rhs._attributes);
		std::swap(_methods, rhs._methods);

		TypeRegistry::add(this);
	}

	Type::~Type()
	{
		if (hash() != 0u)
		{
			TypeRegistry::remove(this);
		}
	}

	void* Type::allocate() const
	{
#if defined(VCL_ARCH_X86) || defined(VCL_ARCH_X64)
		void* obj = _mm_malloc(_size, _alignment);
#else
		void* obj = aligned_alloc(_alignment, _size);
#endif

		return obj;
	}

	void Type::deallocate(void* ptr) const
	{
#if defined(VCL_ARCH_X86) || defined(VCL_ARCH_X64)
		_mm_free(ptr);
#else
		free(ptr);
#endif
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
			{
				return true; // found a match
			}

			if (meta->nrParents() > 0)
			{
				meta = meta->parents()[0];
			} else
			{
				meta = nullptr;
			}
		}
		return false; // no match found
	}

	bool Type::hasAttribute(const stdext::string_view name) const
	{
		const size_t hash = Vcl::Util::StringHash(name.data(), name.length()).hash();

		const auto* attribIt = std::find_if(std::begin(_attributes), std::end(_attributes), [hash](const AttributeBase* attrib) {
			return attrib->hash() == hash;
		});

		if (attribIt != _attributes.end())
		{
			return true;
		}
		if (nrParents() > 0)
		{
			return parents()[0]->hasAttribute(name);
		}

		return false;
	}

	const AttributeBase* Type::attribute(const stdext::string_view name) const
	{
		VclRequire(hasAttribute(name), "Attribute exists.");

		const size_t hash = Vcl::Util::StringHash(name.data(), name.length()).hash();

		const auto* attribIt = std::find_if(_attributes.cbegin(), _attributes.cend(), [hash](const AttributeBase* attrib) {
			return attrib->hash() == hash;
		});

		if (attribIt != _attributes.cend())
		{
			return *attribIt;
		}
		if (nrParents() > 0)
		{
			return parents()[0]->attribute(name);
		}
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
		for (const auto* attr : _attributes)
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
