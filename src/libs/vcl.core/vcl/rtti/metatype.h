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

// VCL
#include <vcl/core/contract.h>
#include <vcl/core/preprocessor.h>
#include <vcl/rtti/metatypebase.h>
#include <vcl/rtti/metatypeconstructor.h>
#include <vcl/rtti/metatypeconstructor.inl>

template<bool Test, int A, int B>
struct select_if
{
	static const int value = B;
};

template<int A, int B>
struct select_if<true, A, B>
{
	static const int value = A;
};

namespace Vcl { namespace RTTI 
{
	// Based on the article series:
	// http://seanmiddleditch.com/journal/2012/01/c-metadata-part-i-singletons-and-lookup/
	template <typename MetaType>
	class MetaTypeSingleton
	{
	public:
		static const Type* get() { Require(_metatype, "Type is initialized."); return _metatype; }

	public:
		template<int N>
		static void allocate(const char (&str)[N], void* allocator)
		{
			Require(_metatype == nullptr, "Type is not initialized.");

			// Allocate a new type object
			//void* obj;

			// Construct the type
			//_metatype = new(obj) ConstructableType<MetaType>;
			size_t alignment = select_if<std::is_abstract<MetaType>::value, 0, alignof(MetaType)>::value;
			_metatype = new ConstructableType<MetaType>(str, Vcl::Util::StringHash(str).hash(), sizeof(MetaType), alignment);
		}

		static void deallocate()
		{
			delete _metatype;
		}

		static Type* construct(ConstructableType<MetaType>* type);

	private:
		static Type* _metatype;
	};

	template<typename T>
	Type* MetaTypeSingleton<T>::_metatype = nullptr;

	// Template specializations matching different type variations
	template <typename MetaType>
	class MetaTypeSingleton<const MetaType> : public MetaTypeSingleton<MetaType> {};

	template <typename MetaType>
	class MetaTypeSingleton<MetaType&> : public MetaTypeSingleton<MetaType> {};

	template <typename MetaType>
	class MetaTypeSingleton<const MetaType&> : public MetaTypeSingleton<MetaType> {};

	template <typename MetaType>
	class MetaTypeSingleton<MetaType&&> : public MetaTypeSingleton<MetaType> {};

	template <typename MetaType>
	class MetaTypeSingleton<MetaType*> : public MetaTypeSingleton<MetaType> {};

	template <typename MetaType>
	class MetaTypeSingleton<const MetaType*> : public MetaTypeSingleton<MetaType> {};
}}

#define VCL_METAOBJECT(name) Vcl::RTTI::MetaTypeSingleton<name>::get()
#define VCL_DECLARE_METAOBJECT(name) public: virtual const Vcl::RTTI::Type* metaType() const { return Vcl::RTTI::MetaTypeSingleton<name>::get(); }
#define VCL_DEFINE_METAOBJECT(name) \
	Vcl::RTTI::Type* Vcl::RTTI::MetaTypeSingleton<name>::construct(Vcl::RTTI::ConstructableType<name>* type)
#define VCL_ALLOC_METAOBJECT(name) Vcl::RTTI::MetaTypeSingleton<name>::allocate(#name, nullptr)
#define VCL_INIT_METAOBJECT(name) Vcl::RTTI::MetaTypeSingleton<name>::construct(static_cast<Vcl::RTTI::ConstructableType<name>*>(const_cast<Vcl::RTTI::Type*>(Vcl::RTTI::MetaTypeSingleton<name>::get())))
