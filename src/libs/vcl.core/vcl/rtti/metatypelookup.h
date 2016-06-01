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
#include <vcl/rtti/metatype.h>

namespace Vcl { namespace RTTI 
{
	// Based on the article series:
	// http://seanmiddleditch.com/journal/2012/01/c-metadata-part-i-singletons-and-lookup/
	
	template <typename MetaType>
	struct MetaIsDynamic
	{
	private:
		struct no_return {};
		template <typename U> static char check(decltype(static_cast<U*>(0)->metaType()));
		template <typename U> static no_return check(...);

	public:
		static const bool value = !std::is_same<no_return, decltype(check<MetaType>(0))>::value;
	};

	template <typename MetaType>
	struct MetaLookup
	{
		template <typename U>
		static typename std::enable_if<MetaIsDynamic<U>::value, const Type*>::type resolve(const U& obj)
		{
			return obj.metaType();
		}

		template <typename U>
		static typename std::enable_if<!MetaIsDynamic<U>::value, const Type*>::type resolve(const U&)
		{
			return MetaTypeSingleton<U>::get();
		}

		static const Type* get(const MetaType& obj)
		{
			return resolve<MetaType>(obj);
		}
	};

	template <typename MetaType>
	struct MetaLookup<MetaType*>
	{
		static const Type* get(const MetaType* obj) { return MetaLookup<MetaType>::get(*obj); }
	};

	template <typename MetaType>
	struct MetaLookup<const MetaType*> : public MetaLookup<MetaType*> {};
}}


template<typename T>
const Vcl::RTTI::Type* vcl_meta_type()
{
	return Vcl::RTTI::MetaTypeSingleton<T>::get();
}

template<typename T>
const Vcl::RTTI::Type* vcl_meta_type(const T& obj)
{
	return Vcl::RTTI::MetaLookup<T>::get(obj);
}

template <typename TargetType, typename InputType>
TargetType* vcl_cast(InputType* input)
{
  const auto* meta = vcl_meta_type(input);
  const auto* target = vcl_meta_type<TargetType>();
  return meta != nullptr && meta->isA(target) ? static_cast<TargetType*>(input) : nullptr;
}

template <typename TargetType, typename InputType>
const TargetType* vcl_cast(const InputType* input)
{
  const auto* meta = vcl_meta_type(input);
  const auto* target = vcl_meta_type<TargetType>();
  return meta != nullptr && meta->isA(target) ? static_cast<const TargetType*>(input) : nullptr;
}
