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
#include <vcl/rtti/metatyperegistry.h>

// VCL
#include <vcl/util/hashedstring.h>

namespace Vcl { namespace RTTI 
{
	// Stores pointer to instatiated type objects.
	// This registry is non-owning. Initialization and cleanup
	// must be performed by the calling code.
	using TypeMap = std::unordered_map<size_t, const Type*>;

	namespace
	{
		TypeMap& instance()
		{
			// Since C++11 this initialization is thread-safe
			static TypeMap types;

			return types;
		}
	}

	void TypeRegistry::add(const Type* meta)
	{
		TypeMap& metas = instance();

		// Only add the entry if it is not yet added
		const auto itr = metas.find(meta->hash());
		if (itr == metas.end())
		{
			metas.emplace(meta->hash(), meta);
		}
	}

	void TypeRegistry::remove(const Type* meta)
	{
		TypeMap& metas = instance();

		// Only remove the type, if it is the one stored
		const auto itr = metas.find(meta->hash());
		if (itr != metas.end() && itr->second == meta)
		{
			metas.erase(itr);
		}
	}

	const Type* TypeRegistry::get(stdext::string_view name)
	{
		const TypeMap& metas = instance();

		// Compute hash
		const size_t hash = Vcl::Util::StringHash(name.data(), name.length()).hash();

		const auto meta = metas.find(hash);
		return (meta == metas.end()) ? nullptr : meta->second;
	}
}}
