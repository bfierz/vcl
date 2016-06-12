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
#include <vcl/rtti/metatypeconstructor.h>

namespace Vcl { namespace RTTI 
{
	// Based on the article series:
	// http://seanmiddleditch.com/journal/2012/01/c-metadata-part-i-singletons-and-lookup/
	template <typename MetaType>
	class MetaTypeSingleton
	{
	public:
		static const Type* get() { return &_metatype; }

	private:
		/*!
		 *	\brief Initialize a new meta-type
		 *	\param str Readable name of the meta-type
		 */
		template<int N>
		static ConstructableType<MetaType> init(const char(&str)[N]);

		/*!
		 *	\brief Configure a newly initialized meta-type
		 *	\param type Meta-type to configure
		 */
		static void construct(ConstructableType<MetaType>* type);

	private:
		//! Instance of the meta-type
		static ConstructableType<MetaType> _metatype;
	};

	template<typename MetaType>
	template<int N>
	ConstructableType<MetaType> MetaTypeSingleton<MetaType>::init(const char(&str)[N])
	{
		ConstructableType<MetaType> type{ str, sizeof(MetaType), alignof(MetaType) };

		// Build the content of the metatype
		construct(&type);

		return std::move(type);
	}

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
#define VCL_DECLARE_METAOBJECT(name) public: virtual const Vcl::RTTI::Type* metaType() const;
#define VCL_DEFINE_METAOBJECT(name) \
	namespace Vcl { namespace RTTI {                                          \
	template <>                                                               \
	class MetaTypeSingleton<name>                                             \
	{                                                                         \
	public:                                                                   \
		static const Type* get() { return &_metatype; }                       \
	private:                                                                  \
		template<int N>                                                       \
		static ConstructableType<name> init(const char(&str)[N])              \
		{                                                                     \
			ConstructableType<name> type{ str, sizeof(name), alignof(name) }; \
			construct(&type);                                                 \
			return std::move(type);                                           \
		}                                                                     \
		static void construct(ConstructableType<name>* type);                 \
	private:                                                                  \
		static ConstructableType<name> _metatype;                             \
	};}}                                                                      \
	Vcl::RTTI::ConstructableType<name> Vcl::RTTI::MetaTypeSingleton<name>::_metatype = Vcl::RTTI::MetaTypeSingleton<name>::init(#name); \
	const Vcl::RTTI::Type* name::metaType() const { return Vcl::RTTI::MetaTypeSingleton<name>::get(); } \
	void Vcl::RTTI::MetaTypeSingleton<name>::construct(Vcl::RTTI::ConstructableType<name>* type)
