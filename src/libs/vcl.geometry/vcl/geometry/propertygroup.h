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

// C++ standard libary
#include <memory>
#include <unordered_map>

// VCL
#include <vcl/geometry/property.h>

namespace Vcl { namespace Geometry
{
	template<typename IndexT>
	class PropertyGroup
	{
	public:
		using map_type   = std::unordered_map<std::string, std::unique_ptr<PropertyBase>>;
		using index_type = IndexT;

	public:
		PropertyGroup(const std::string& name)
		: _name(name)
		, _propertySize(0)
		, _propertyAllocation(0)
		{
		}
		
		PropertyGroup(const PropertyGroup<IndexT>& other)
		: _name(other._name)
		, _propertySize(other._propertySize)
		, _propertyAllocation(other._propertyAllocation)
		{
			for (const auto& entry : other._data)
			{
				_data[entry.first] = entry.second->clone();
			}
		}

		~PropertyGroup()
		{
			removeAll();
		}

		PropertyGroup& operator = (const PropertyGroup<IndexT>& other)
		{
			if (this != &other) // Protect against invalid self-assignment
			{
				_name = other._name;
				_propertySize = other._propertySize;
	 
				for (const auto& entry : other._data)
				{
					_data[entry.first] = entry.second->clone();
				}
			}

			// By convention, always return *this
			return *this;
		}

	public:
		void clear()
		{
			for (const auto& entry : _data)
			{
				entry.second->clear();
			}
		}

		void removeAll()
		{
			_data.clear();
		}

	public:
		template<typename T>
		Property<T, index_type>* add(const std::string& name, typename Property<T, index_type>::rvalue_reference init_value = typename Property<T, index_type>::value_type())
		{		
			if (_data.find(name) == _data.end())
			{
				auto data = std::make_unique<Property<T, index_type>>(name, std::forward<typename Property<T, index_type>::value_type>(init_value));

				data->resize(_propertySize);
				_data.emplace(name, std::move(data));
			}

			return static_cast<Property<T, index_type>*>(_data[name].get());
		}

		template<typename T>
		Property<T, index_type>* add(const std::string& name, typename Property<T, index_type>::const_reference init_value)
		{
			if (_data.find(name) == _data.end())
			{
				auto data = std::make_unique<Property<T, index_type>>(name, init_value);

				data->resize(_propertySize);
				_data.emplace(name, std::move(data));
			}

			return static_cast<Property<T, index_type>*>(_data[name].get());
		}

		void add(std::unique_ptr<PropertyBase> prop)
		{
			VclRequire(_data.find(prop->name()) == _data.end(), "Property must not exist.");

			if (_data.find(prop->name()) == _data.end())
			{
				prop->resize(_propertySize);

				const auto& name = prop->name();
				_data.emplace(name, std::move(prop));
			}
		}

		void remove(const std::string& name)
		{
			auto elem = _data.find(name);
			if (elem != _data.end())
			{
				elem->second->clear();
				VCL_SAFE_DELETE(elem->second);

				_data.erase(elem);
			}
		}

		template<typename T>
		Property<T, index_type>* property(const std::string& name, bool create_if_not_found = false, typename Property<T, index_type>::rvalue_reference init_value = typename Property<T, index_type>::value_type())
		{
			auto prop = _data.find(name);
			if (prop != _data.end())
			{
				return static_cast<Property<T, index_type>*>(prop->second.get());
			}
			else if (create_if_not_found)
			{
				return add<T>(name, init_value);
			}

			return nullptr;
		}
		
		template<typename T>
		const Property<T, index_type>* property(const std::string& name) const
		{
			auto prop = _data.find(name);
			if (prop != _data.end())
			{
				return static_cast<const Property<T, index_type>*>(prop->second);
			}

			return nullptr;
		}
		
		const PropertyBase* propertyBase(const std::string& name) const
		{
			auto prop = _data.find(name);
			if (prop != _data.end())
			{
				return prop->second;
			}

			return nullptr;
		}

		bool exists(const std::string& name)
		{
			auto prop = _data.find(name);
			if (prop != _data.end())
				return true;
			else
				return false;
		}

		size_t propertySize() const
		{
			return _propertySize;
		}

		void reserveProperties(size_t size)
		{
			for (const auto& entry : _data)
				entry.second->reserve(size);
			
			_propertyAllocation = size;
		}

		void resizeProperties(size_t size)
		{
			for (const auto& entry : _data)
				entry.second->resize(size);
			
			_propertySize = size;
			_propertyAllocation = std::max(_propertyAllocation, _propertySize);
		}

	private:
		//! Data of the single properties
		map_type _data;

		//! Name of the property group
		std::string _name;

		//! Size of the properties
		size_t _propertySize;

		//! Allocated storage for each property
		size_t _propertyAllocation;
	};
}}
