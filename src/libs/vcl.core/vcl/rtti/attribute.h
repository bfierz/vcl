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
#include <functional>
#include <memory>
#include <string>

// VCL
#include <vcl/core/3rdparty/any.hpp>
#include <vcl/core/contract.h>
#include <vcl/core/convert.h>
#include <vcl/rtti/attributebase.h>
#include <vcl/rtti/factory.h>
#include <vcl/rtti/metatypelookup.h>
#include <vcl/rtti/serializer.h>

namespace Vcl { namespace RTTI 
{
	template<typename MetaType, typename T>
	class Attribute : public AttributeBase
	{
	public:
		Attribute(const char* name, const T& (MetaType::*getter)() const, void (MetaType::*setter)(const T&))
		: AttributeBase(name)
		, _getter(std::mem_fn(getter))
		, _setter(std::mem_fn(setter))
		{
			if (setter != nullptr)
				setHasSetter();
			if (getter != nullptr)
				setHasGetter();
		}
		Attribute(const char* name, T (MetaType::*getter)() const, void (MetaType::*setter)(T))
		: AttributeBase(name)
		, _getter(std::mem_fn(getter))
		, _setter(std::mem_fn(setter))
		{
			if (setter != nullptr)
				setHasSetter();
			if (getter != nullptr)
				setHasGetter();
		}
		
	public:
		const T& get(const MetaType& obj) const
		{
			return _getter(obj);
		}

		void set(MetaType& obj, const T& val)
		{
			_setter(obj, val);
		}

	public:
		virtual void set(void* object, const linb::any& param) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(param);
			DebugError("Not implemented.");
		}
		virtual void set(void* object, const std::string& param) const override
		{
			Require(_setter, "Setter is valid.");

			_setter(*static_cast<MetaType*>(object), from_string<T>(param));
		}
		virtual void get(void* object, void* param, void* result) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(param);
			VCL_UNREFERENCED_PARAMETER(result);
			DebugError("Not implemented.");

			//_getter()
		}
		virtual void get(void* object, const std::string& param, void* result) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(param);
			VCL_UNREFERENCED_PARAMETER(result);
			DebugError("Not implemented.");
		}

		virtual void serialize(Serializer& ser, const void* object) override
		{
			const auto& val = get(*static_cast<const MetaType*>(object));
			std::string str = to_string(val);

			ser.writeAttribute(name(), str);
		}

		virtual void deserialize(Deserializer& deser, void* object) override
		{
			Require(deser.hasAttribute(name()), "Attribute is available.");

			set(object, deser.readAttribute(name()));
		}

	private:
		//! Function pointer to the stored getter
		std::function<const T& (const MetaType&)> _getter;

		//! Function pointer to the stored setter
		std::function<void (MetaType&, const T&)> _setter;
	};
	
	template<typename MetaType, typename T>
	class Attribute<MetaType, std::unique_ptr<T>> : public AttributeBase
	{
		using  AttrT = std::unique_ptr<T>;

	public:
		Attribute(const char* name, T* (MetaType::*getter)() const, void (MetaType::*setter)(AttrT))
		: AttributeBase(name)
		, _getter(std::mem_fn(getter))
		, _setter(std::mem_fn(setter))
		{
			if (setter != nullptr)
				setHasSetter();
			if (getter != nullptr)
				setHasGetter();
		}
		
	public:
		T* get(const MetaType& obj) const
		{
			return _getter(obj);
		}

		void set(MetaType& obj, AttrT val)
		{
			_setter(obj, std::move(val));
		}

	public:
		virtual void set(void* object, const linb::any& param) const override
		{
			Require(object, "Object is set.");

			auto ptr = linb::any_cast<T*>(param);
			_setter(*static_cast<MetaType*>(object), std::unique_ptr<T>(ptr));
		}
		virtual void set(void* object, const std::string& param) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(param);

			DebugError("Not implemented.");
		}
		virtual void get(void* object, void* param, void* result) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(param);
			VCL_UNREFERENCED_PARAMETER(result);

			DebugError("Not implemented.");
		}
		virtual void get(void* object, const std::string& param, void* result) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(param);
			VCL_UNREFERENCED_PARAMETER(result);

			DebugError("Not implemented.");
		}

		virtual void serialize(Serializer& ser, const void* object) override
		{
			// Print attribute
			ser.writeAttribute(name(), "");

			// Write content of the attribute
			auto* type = vcl_meta_type<T>();
			type->serialize(ser, object);
		}

		virtual void deserialize(Deserializer& deser, void* object) override
		{
			Require(deser.hasAttribute(name()), "Attribute is available.");

			// Start reading a new object
			deser.beginType(name());

			// Read content of the attribute
			auto type = vcl_meta_type(deser.readType());
			auto store = (MetaType*) Factory::create(deser.readType().c_str());
			auto val = std::unique_ptr<T>(store);

			type->deserialize(deser, val.get());

			auto& obj = *static_cast<MetaType*>(object);
			set(obj, std::move(val));

			// Done reading the type
			deser.endType();
		}

	private:
		/// Function pointer to the stored getter
		std::function<T* (const MetaType&)> _getter;

		/// Function pointer to the stored setter
		std::function<void (MetaType&, AttrT)> _setter;
	};
	
	/*template<typename MetaType, typename T>
	class Attribute<MetaType, std::shared_ptr<T>> : public AttributeBase
	{
		typedef T ValueType;
		typedef std::shared_ptr<T> AttrT;

	public:
		Attribute(const char* name, AttrT (MetaType::*getter)() const, void (MetaType::*setter)(AttrT))
		: AttributeBase(name)
		, _getter(std::mem_fn(getter))
		, _setter(std::mem_fn(setter))
		{
			setIsReference();
			setIsShared();
			if (setter != nullptr)
				setHasSetter();
			if (getter != nullptr)
				setHasGetter();
		}

		virtual ~Attribute()
		{
		}
		
	public:
		AttrT get(const MetaType& obj) const
		{
			return _getter(obj);
		}

		void set(MetaType& obj, AttrT val)
		{
			_setter(obj, val);
		}

	public:
		virtual void set(void* object, const linb::any& param) const override
		{
			Require(object, "Object is set.");
			//Require(param, "Value is set.");

			auto shared = boost::any_cast<AttrT>(&param);
			if (shared)
			{
				_setter(*static_cast<MetaType*>(object), *shared);
			}
			else
			{
				ValueType* val = reinterpret_cast<ValueType*>(boost::any_cast<void*>(param));
				_setter(*static_cast<MetaType*>(object), AttrT(val));
			}
		}
		virtual void set(void* object, const std::string& param) const override
		{
			// This method could be implemented by looking up an object in a data base
			DebugError("Not implemented.");
		}
		virtual void get(void* object, void* param, void* result) const override
		{
			DebugError("Not implemented.");
		}
		virtual void get(void* object, const std::string& param, void* result) const override
		{
			DebugError("Not implemented.");
		}

	private:
		/// Function pointer to the stored getter
		std::function<AttrT (const MetaType&)> _getter;

		/// Function pointer to the stored setter
		std::function<void (MetaType&, AttrT)> _setter;
	};
	
	template<typename MetaType, typename T>
	class Attribute<MetaType, T*> : public AttributeBase
	{
		typedef T* AttrT;

	public:
		Attribute(const char* name, AttrT (MetaType::*getter)() const, void (MetaType::*setter)(AttrT))
		: AttributeBase(name)
		, _getter(std::mem_fn(getter))
		, _setter(std::mem_fn(setter))
		{
			setIsReference();
			if (setter != nullptr)
				setHasSetter();
			if (getter != nullptr)
				setHasGetter();
		}

		virtual ~Attribute()
		{
		}
		
	public:
		AttrT get(const MetaType& obj) const
		{
			return _getter(obj);
		}

		void set(MetaType& obj, AttrT val)
		{
			_setter(obj, val);
		}

	public:
		virtual void set(void* object, const linb::any& param) const override
		{
			Require(object, "Object is set.");
			//Require(param, "Value is set.");

			AttrT obj_ptr = boost::any_cast<T*>(param);

			_setter(*static_cast<MetaType*>(object), obj_ptr);
		}
		virtual void set(void* object, const std::string& param) const override
		{
			DebugError("Not implemented.");
		}
		virtual void get(void* object, void* param, void* result) const override
		{
			DebugError("Not implemented.");
		}
		virtual void get(void* object, const std::string& param, void* result) const override
		{
			DebugError("Not implemented.");
		}

	private:
		/// Function pointer to the stored getter
		std::function<AttrT (const MetaType&)> _getter;

		/// Function pointer to the stored setter
		std::function<void (MetaType&, AttrT)> _setter;
	};*/
}}
