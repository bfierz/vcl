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
#include <initializer_list>
#include <vector>

// VCL
#include <vcl/core/any.h>
#include <vcl/core/preprocessor.h>
#include <vcl/rtti/metatypebase.h>

#define VCL_RTTI_BASES(Object, ...) auto VCL_PP_JOIN(Object, _parents) = vcl_meta_types<__VA_ARGS__>();
#define VCL_RTTI_REGISTER_BASES(Object) type->registerBaseClasses(VCL_PP_JOIN(Object, _parents));

namespace Vcl { namespace RTTI 
{
	template<typename T>
	class ConstructableType : public Type
	{
		using MetaType = T;

	public:
		template<size_t N>
		ConstructableType(const char(&name)[N], size_t size, size_t alignment)
		: Type(name, size, alignment)
		{
		}
		ConstructableType(ConstructableType&& rhs) = default;

	public:		
		template<size_t N>
		void registerBaseClasses(std::array<const Type*, N>& bases)
		{
			_parents = bases;
		}

		template<size_t N>
		void registerConstructors(std::array<const ConstructorBase*, N>& constructors)
		{
			_constructors.set(constructors);
		}

		template<size_t N>
		void registerAttributes(std::array<const AttributeBase*, N>& attributes)
		{
			_attributes = attributes;
		}

		virtual void destruct(void* ptr) const override
		{
			VCL_UNREFERENCED_PARAMETER(ptr);
			static_cast<T*>(ptr)->~T();
		}
	};

	template<typename T>
	class DynamicConstructableType : public Type
	{
		using MetaType = T;

	public:
		template<size_t N>
		DynamicConstructableType(const char(&name)[N], size_t size, size_t alignment)
		: Type(name, size, alignment)
		{
		}
		DynamicConstructableType(DynamicConstructableType&& rhs) = default;

	public:
		template<typename Args>
		DynamicConstructableType<T>* inherit();

		DynamicConstructableType<T>* addConstructor();
		
		template<typename... Args>
		DynamicConstructableType<T>* addConstructor(Parameter<Args>... descriptors);

		template<size_t N, typename AttribT>
		DynamicConstructableType<T>* addAttribute(const char(&name)[N], AttribT(MetaType::*getter)() const, void(MetaType::*setter)(AttribT));

		template<size_t N, typename AttribT>
		DynamicConstructableType<T>* addAttribute(const char(&name)[N], AttribT*(MetaType::*getter)() const, void(MetaType::*setter)(std::unique_ptr<AttribT>));

		template<size_t N, typename AttribT>
		DynamicConstructableType<T>* addAttribute(const char(&name)[N], const AttribT& (MetaType::*getter)() const, void (MetaType::*setter)(const AttribT&));

		virtual void destruct(void* ptr) const override
		{
			static_cast<T*>(ptr)->~T();
		}

	private:
		std::vector<const Type*> _concreteParents;
		std::vector<std::unique_ptr<ConstructorBase>> _concreteConstructors;
		std::vector<std::unique_ptr<AttributeBase>> _concreteAttributes;
	};
}}
