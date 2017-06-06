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

 // Foonathan Allocator library
VCL_BEGIN_EXTERNAL_HEADERS
#include <foonathan/memory/container.hpp>
#include <foonathan/memory/memory_pool.hpp>
#include <foonathan/memory/namespace_alias.hpp>
#include <foonathan/memory/std_allocator.hpp>
VCL_END_EXTERNAL_HEADERS

// VCL
#include <vcl/util/hashedstring.h>

namespace Vcl { namespace RTTI 
{
	using TypeMap = std::unordered_map<size_t, const Type*, std::hash<size_t>,
		std::equal_to<size_t>, memory::std_allocator<std::pair<const size_t, const Type*>, memory::memory_pool<>>>;

	namespace
	{
		TypeMap& instance()
		{
			using namespace memory::literals;

			// a memory pool RawAllocator
			static memory::memory_pool<> pool{ memory::unordered_map_node_size<std::pair<const size_t, const Type*>>::value, 4_KiB };

			// Since C++11 this initialization is thread-safe
			static TypeMap types{ pool };

			return types;
		}
	}

	void TypeRegistry::add(const Type* meta)
	{
		TypeMap& metas = instance();

		// Only add the entry if it is not yet added
		auto itr = metas.find(meta->hash());
		if (itr == metas.end())
			metas.emplace(meta->hash(), meta);
	}
	
	void TypeRegistry::remove(const Type* meta)
	{
		TypeMap& metas = instance();

		// Only remove the type, if it is the one stored
		auto itr = metas.find(meta->hash());
		if (itr != metas.end() && itr->second == meta)
			metas.erase(itr);
	}

	const Type* TypeRegistry::get(const gsl::cstring_span<> name)
	{
		const TypeMap& metas = instance();

		// Compute hash
		size_t hash = Vcl::Util::StringHash(name).hash();

		auto meta = metas.find(hash);
		return (meta == metas.end()) ? nullptr : meta->second;
	}
}}
