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

// VCL
#include <vcl/rtti/attribute.h>
#include <vcl/rtti/constructor.h>
#include <vcl/rtti/metatypeconstructor.h>
#include <vcl/rtti/metatypelookup.h>

namespace Vcl { namespace RTTI 
{
	template<typename T>
	template<typename Args>
	DynamicConstructableType<T>* DynamicConstructableType<T>::inherit()
	{
		_concreteParents.push_back(vcl_meta_type<Args>());
		_parents = _concreteParents;

		return this;
	}

	template<typename T>
	DynamicConstructableType<T>* DynamicConstructableType<T>::addConstructor()
	{
		// Create an new constructor
		auto constr = std::make_unique<Constructor<T>>();

		// Store the constructor in the table
		_concreteConstructors.push_back(std::move(constr));
		_constructors.set(_concreteConstructors);

		return this;
	}

	template<typename T>
	template<typename... Args>
	DynamicConstructableType<T>* DynamicConstructableType<T>::addConstructor(Parameter<Args>... descriptors)
	{
		// Create an new constructor
		auto constr = std::make_unique<Constructor<T, Args...>>(descriptors...);

		// Store the constructor in the table
		_concreteConstructors.push_back(std::move(constr));
		_constructors.set(_concreteConstructors);

		return this;
	}

	template<typename T>
	template<size_t N, typename AttribT>
	DynamicConstructableType<T>* DynamicConstructableType<T>::addAttribute(const char(&name)[N], AttribT(MetaType::*getter)() const, void(MetaType::*setter)(AttribT))
	{
		auto attrib = std::make_unique<Attribute<T, AttribT>>(name, getter, setter);

		_concreteAttributes.push_back(std::move(attrib));
		_attributes = { (const AttributeBase**)_concreteAttributes.data(), (std::ptrdiff_t) _concreteAttributes.size() };

		return this;
	}

	template<typename T>
	template<size_t N, typename AttribT>
	DynamicConstructableType<T>* DynamicConstructableType<T>::addAttribute(const char(&name)[N], AttribT*(MetaType::*getter)() const, void(MetaType::*setter)(std::unique_ptr<AttribT>))
	{
		auto attrib = std::make_unique<Attribute<T, std::unique_ptr<AttribT>>>(name, getter, setter);

		_concreteAttributes.push_back(std::move(attrib));
		_attributes = { (const AttributeBase**)_concreteAttributes.data(), (std::ptrdiff_t) _concreteAttributes.size() };

		return this;
	}

	template<typename T>
	template<size_t N, typename AttribT>
	DynamicConstructableType<T>* DynamicConstructableType<T>::addAttribute(const char(&name)[N], const AttribT& (MetaType::*getter)() const, void (MetaType::*setter)(const AttribT&))
	{
		auto attrib = std::make_unique<Attribute<T, const AttribT&>>(name, getter, setter);

		_concreteAttributes.push_back(std::move(attrib));
		_attributes = { (const AttributeBase**)_concreteAttributes.data(), static_cast<std::ptrdiff_t>(_concreteAttributes.size()) };

		return this;
	}
}}
