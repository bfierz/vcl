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

#include "any.h"

// VCL
#include <vcl/rtti/metatypebase.h>

namespace Vcl { namespace RTTI 
{
	namespace Internal
	{
		template<typename T>
		ParameterBase* copy(const Parameter<T>& orig)
		{
			return new Parameter<T>(orig);
		}
	}

	template<typename T>
	class ConstructableType : public Type
	{
		typedef typename T MetaType;

	public:
		ConstructableType(const char* name, size_t size, size_t alignment)
		: Type(name, size, alignment)
		{
		}
		ConstructableType(const char* name, size_t hash, size_t size, size_t alignment)
		: Type(name, hash, size, alignment)
		{
		}
		ConstructableType(ConstructableType&& rhs)
		: Type(std::move(rhs))
		{
			std::swap(_init, rhs._init);
		}
		~ConstructableType()
		{
		}

	public:
		template<typename Args>
		ConstructableType<T>* inherit();
		
		ConstructableType<T>* addConstructor();

		template<typename... Args>
		ConstructableType<T>* addConstructor(Parameter<Args>... descriptors)
		{
			// Create an new constructor
			std::array<ParameterBase*, sizeof...(Args)> meta_data{ { Internal::copy(descriptors)... } };
			auto constr = std::make_unique<Constructor<T, Args...>>(meta_data);

			// Store the constructor in the table
			_constructors.add(std::move(constr));

			return this;
		}

		template<typename AttribT>
		ConstructableType<T>* addAttribute(const char* name, AttribT(MetaType::*getter)() const, void(MetaType::*setter)(AttribT));

		template<typename AttribT>
		ConstructableType<T>* addAttribute(const char* name, AttribT*(MetaType::*getter)() const, void(MetaType::*setter)(std::unique_ptr<AttribT>));

		template<typename AttribT>
		ConstructableType<T>* addAttribute(const char* name, const AttribT& (MetaType::*getter)() const, void (MetaType::*setter)(const AttribT&))
		{
			auto attrib = new Attribute<T, AttribT>(name, getter, setter);
			
			_attributes.push_back(std::move(attrib));

			return this;
		}

		ConstructableType<T>* createFactory()
		{
			Require(_isConstructable == false, "Constructor not set.");

			_isConstructable = true;

			_init = [&] (void* ptr, const std::initializer_list<cdiggins::any>& params) -> void { new(ptr) T; };
			
			return this;
		}

		template<typename T0>
		ConstructableType<T>* createFactory()
		{
			Require(_isConstructable == false, "Constructor not set.");

			_isConstructable = true;

			_init = [] (void* ptr, const std::initializer_list<cdiggins::any>& params) -> void
			{
				auto* param_list = params.begin();
				cdiggins::any a0 = param_list[0];

				const T0* p0 = cdiggins::unsafe_any_cast<T0>(&a0);

				new(ptr) T(*p0);
			};
			
			return this;
		}

		template<typename T0, typename T1>
		ConstructableType<T>* createFactory()
		{
			Require(_isConstructable == false, "Constructor not set.");
			
			typedef void function_t(void*);

			_isConstructable = true;
			
			_init = [&] (void* ptr, const std::initializer_list<cdiggins::any>& params) -> void
			{
				auto* param_list = params.begin();

				new(ptr) T(cdiggins::any_cast<T0>(param_list[0]), cdiggins::any_cast<T1>(param_list[1]));
			};
			
			return this;
		}

	private:
		virtual void construct(void* ptr, const std::initializer_list<cdiggins::any>& params) const
		{
			if (_init)
			{
				_init(ptr, params);
			}
		}
		
		virtual void destruct(void* ptr) const override
		{
			static_cast<T*>(ptr)->~T();
		}

	private:
		std::function<void(void*, const std::initializer_list<cdiggins::any>&)> _init;
	};
}}
