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
#include <vcl/core/container/array.h>
#include <vcl/core/any.h>
#include <vcl/core/contract.h>
#include <vcl/core/convert.h>
#include <vcl/core/enum.h>
#include <vcl/rtti/attributebase.h>
#include <vcl/rtti/factory.h>
#include <vcl/rtti/metatypelookup.h>
#include <vcl/rtti/serializer.h>

#define VCL_RTTI_ATTR_TABLE_BEGIN(Object) namespace { auto VCL_PP_JOIN(Object, _attributes) = std::make_tuple(
#define VCL_RTTI_ATTR(Object, name, type, getter, setter) Vcl::RTTI::Attribute<Object, type>{ name, &Object::getter, &Object::setter }
#define VCL_RTTI_ATTR_TABLE_END(Object) ); auto VCL_PP_JOIN(Object, _attribute_bases) = Vcl::Core::make_array_from_tuple<const Vcl::RTTI::AttributeBase*>(VCL_PP_JOIN(Object, _attributes)); }
#define VCL_RTTI_REGISTER_ATTRS(Object) type->registerAttributes(VCL_PP_JOIN(Object, _attribute_bases));

namespace Vcl { namespace RTTI 
{
	template<typename T>
	class EnumAttribute : public EnumAttributeBase
	{
	public:
		template<size_t N>
		VCL_CPP_CONSTEXPR_14 EnumAttribute(const char(&name)[N])
			: EnumAttributeBase(name)
		{
		}

		uint32_t count() const override
		{
			return Vcl::enumCount<T>();
		}
		uint32_t enumValue(uint32_t i) const override
		{
			return static_cast<uint32_t>(Vcl::enumValue<T>(i));
		}
		std::string enumName(uint32_t i) const override
		{
			return Vcl::enumName<T>(i);
		}
	};

	template<typename MetaType, typename T>
	class Attribute : public std::conditional<std::is_enum<T>::value, EnumAttribute<T>, AttributeBase>::type
	{
		using Base = typename std::conditional<std::is_enum<T>::value, EnumAttribute<T>, AttributeBase>::type;
	public:
		using Getter = T (MetaType::*)() const;
		using Setter = void (MetaType::*)(T);

	public:
		template<size_t N>
		VCL_CPP_CONSTEXPR_14 Attribute(const char(&name)[N], Getter getter, Setter setter)
		: Base(name)
		, _getter(getter)
		, _setter(setter)
		{
			if (setter != nullptr)
				this->setHasSetter();
			if (getter != nullptr)
				this->setHasGetter();
		}
		
	public:
		T get(const MetaType& obj) const
		{
			VclRequire(_getter, "Getter is valid.");
			return (obj.*_getter)();
		}

		void set(MetaType& obj, T val) const
		{
			VclRequire(_setter, "Setter is valid.");
			(obj.*_setter)(std::move(val));
		}

	public:
		virtual void set(void* object, const std::any& param) const override
		{
			VclRequire(_setter, "Setter is valid.");

			set(*static_cast<MetaType*>(object), std::any_cast<T>(param));
		}
		virtual void set(void* object, const std::string& param) const override
		{
			VclRequire(_setter, "Setter is valid.");

			set(*static_cast<MetaType*>(object), from_string<T>(param));
		}
		virtual void get(const void* object, std::any& result) const override
		{
			VclRequire(_getter, "Getter is valid.");

			const auto* obj = static_cast<const MetaType*>(object);
			result = get(*obj);
		}
		virtual void get(const void* object, std::string& result) const override
		{
			VclRequire(_getter, "Getter is valid.");

			const auto* obj = static_cast<const MetaType*>(object);
			result = to_string(get(*obj));
		}

		virtual void serialize(Serializer& ser, const void* object) const override
		{
			T val = get(*static_cast<const MetaType*>(object));
			std::string str = to_string(val);

			ser.writeAttribute(this->name(), str);
		}

		virtual void deserialize(Deserializer& deser, void* object) const override
		{
			VclRequire(deser.hasAttribute(this->name()), "Attribute is available.");

			set(object, deser.readAttribute(this->name()));
		}

	private:
		//! Function pointer to the stored getter
		Getter _getter;

		//! Function pointer to the stored setter
		Setter _setter;
	};

	template<typename MetaType, typename T>
	class Attribute<MetaType, const T&> : public std::conditional<std::is_enum<T>::value, EnumAttribute<T>, AttributeBase>::type
	{
		using Base = typename std::conditional<std::is_enum<T>::value, EnumAttribute<T>, AttributeBase>::type;
	public:
		using  AttrT = const T&;

		using Getter = AttrT (MetaType::*)() const;
		using Setter = void (MetaType::*)(AttrT);

	public:
		template<size_t N>
		VCL_CPP_CONSTEXPR_14 Attribute(const char(&name)[N], Getter getter, Setter setter)
		: Base(name)
		, _getter(getter)
		, _setter(setter)
		{
			if (setter != nullptr)
				this->setHasSetter();
			if (getter != nullptr)
				this->setHasGetter();
		}

	public:
		const T& get(const MetaType& obj) const
		{
			return (obj.*_getter)();
		}

		void set(MetaType& obj, const T& val) const
		{
			(obj.*_setter)(std::move(val));
		}

	public:
		virtual void set(void* object, const std::any& param) const override
		{
			VclRequire(_setter, "Setter is valid.");

			set(*static_cast<MetaType*>(object), std::any_cast<T>(param));
		}
		virtual void set(void* object, const std::string& param) const override
		{
			VclRequire(_setter, "Setter is valid.");

			set(*static_cast<MetaType*>(object), from_string<T>(param));
		}
		virtual void get(const void* object, std::any& result) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(result);
			VclDebugError("Not implemented.");

			//_getter()
		}
		virtual void get(const void* object, std::string& result) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(result);
			VclDebugError("Not implemented.");
		}

		virtual void serialize(Serializer& ser, const void* object) const override
		{
			const auto& val = get(*static_cast<const MetaType*>(object));
			std::string str = to_string(val);

			ser.writeAttribute(this->name(), str);
		}

		virtual void deserialize(Deserializer& deser, void* object) const override
		{
			VclRequire(deser.hasAttribute(this->name()), "Attribute is available.");

			set(object, deser.readAttribute(this->name()));
		}

	private:
		//! Function pointer to the stored getter
		Getter _getter;

		//! Function pointer to the stored setter
		Setter _setter;
	};

	template<typename MetaType, typename T>
	class Attribute<MetaType, std::unique_ptr<T>> : public AttributeBase
	{
		using  AttrT = std::unique_ptr<T>;

		using Getter = T*(MetaType::*)() const;
		using Setter = void (MetaType::*)(AttrT);

	public:
		template<size_t N>
		VCL_CPP_CONSTEXPR_14 Attribute(const char(&name)[N], Getter getter, Setter setter)
		: AttributeBase(name)
		, _getter(getter)
		, _setter(setter)
		{
			if (setter != nullptr)
				this->setHasSetter();
			if (getter != nullptr)
				this->setHasGetter();
		}
		
	public:
		T* get(const MetaType& obj) const
		{
			return (obj.*_getter)();
		}

		void set(MetaType& obj, AttrT val) const
		{
			(obj.*_setter)(std::move(val));
		}

	public:
		virtual void set(void* object, const std::any& param) const override
		{
			VclRequire(object, "Object is set.");

			auto ptr = std::any_cast<T*>(param);
			(static_cast<MetaType*>(object)->*_setter)(AttrT(ptr));
		}
		virtual void set(void* object, const std::string& param) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(param);

			VclDebugError("Not implemented.");
		}
		virtual void get(const void* object, std::any& result) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(result);

			VclDebugError("Not implemented.");
		}
		virtual void get(const void* object, std::string& result) const override
		{
			VCL_UNREFERENCED_PARAMETER(object);
			VCL_UNREFERENCED_PARAMETER(result);

			VclDebugError("Not implemented.");
		}

		virtual void serialize(Serializer& ser, const void* object) const override
		{
			// Write attribute name
			ser.writeAttribute(this->name(), "");

			// Write content of the attribute
			auto* type = vcl_meta_type<T>();
			type->serialize(ser, object);
		}

		virtual void deserialize(Deserializer& deser, void* object) const override
		{
			VclRequire(deser.hasAttribute(this->name()), "Attribute is available.");

			// Start reading a new object
			deser.beginType(this->name());

			// Read content of the attribute
			auto type = vcl_meta_type_by_name(deser.readType());
			auto store = (MetaType*) Factory::create(deser.readType());
			auto val = AttrT(store);

			type->deserialize(deser, val.get());

			auto& obj = *static_cast<MetaType*>(object);
			set(obj, std::move(val));

			// Done reading the type
			deser.endType();
		}

	private:
		/// Function pointer to the stored getter
		Getter _getter;

		/// Function pointer to the stored setter
		Setter _setter;
	};
}}
