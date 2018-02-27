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
#include <memory>
#include <type_traits>

// VCL
#include <vcl/core/memory/allocator.h>
#include <vcl/core/contract.h>

namespace Vcl { namespace Geometry
{
	class PropertyBase
	{
	public:
		inline PropertyBase(const std::string& name)
		: _name(name), _size(0), _allocated(0)
		{}

		inline PropertyBase(const PropertyBase& rhs)
		: _name(rhs._name), _size(rhs._size), _allocated(rhs._allocated)
		{}

		inline PropertyBase(PropertyBase&& rhs)
		: _name(""), _size(0), _allocated(0)
		{
			std::swap(_name, rhs._name);
			std::swap(_size, rhs._size);
			std::swap(_allocated, rhs._allocated);
		}

		inline virtual ~PropertyBase() = default;

	public:
		virtual std::unique_ptr<PropertyBase> clone() const = 0;
		virtual std::unique_ptr<PropertyBase> create() const = 0;

	public:
		inline const std::string& name() const { return _name; }
		inline size_t size() const { return _size; }
		inline size_t allocatedSpace() const { return _allocated; }

	public:
		virtual void clear() = 0;
		virtual void resize(size_t size) = 0;
		virtual void reserve(size_t size) = 0;

	protected:
		inline void setSize(size_t size) { _size = size; }

	private:
		std::string _name;
		size_t _size;

	protected:
		size_t _allocated;
	};

	namespace Internal
	{
		template<typename T>
		struct PropertyMemberType
		{
			typedef T type;
		};
		
		template<>
		struct PropertyMemberType<bool>
		{
			typedef unsigned int type;
		};
	}

	template<typename T, typename IndexT>
	class Property : public PropertyBase
	{
	public:
		typedef typename Internal::PropertyMemberType<T>::type	value_type;

		typedef value_type*			pointer;
		typedef value_type&			reference;
		typedef value_type&&		rvalue_reference;
		typedef const value_type&	const_reference;
		typedef IndexT				index_type;

	public:
		//! Constructor
		Property(const std::string& name, const_reference init_value = value_type())
		: PropertyBase(name)
		, _data(nullptr)
		, _defaultValue(init_value)
		, _allocPolicy(nullptr)
		{
			_allocPolicy = std::make_unique<Core::AlignedAllocPolicy<value_type, 32>>();
			
			VclRequire(size() <= _allocated, "Used size is smaller/equal to allocated size.");
		}

		//! Copy constructor
		Property(const Property& rhs)
		: PropertyBase(rhs)
		{
			_allocPolicy = rhs._allocPolicy->clone();
			_data = _allocPolicy->allocate(rhs.size());

			for (size_t i = 0; i < size(); i++)
				_data[i] = rhs._data[i];
			
			VclRequire(size() <= _allocated, "Used size is smaller/equal to allocated size.");
		}

		//! Move constructor
		Property(Property&& rhs)
		: PropertyBase(std::move(rhs))
		, _data(nullptr)
		, _defaultValue(rhs._defaultValue)
		, _allocPolicy(nullptr)
		{
			std::swap(_allocPolicy, rhs._allocPolicy);
			std::swap(_data, rhs._data);
			std::swap(_defaultValue, rhs._defaultValue);

			VclEnsure(size() <= _allocated, "Used size is smaller/equal to allocated size.");
		}

		virtual ~Property()
		{
			clear();

			if (_allocPolicy)
			{
				_allocPolicy->deallocate(_data, _allocated);
			}
			_allocated = 0;
		}

	public:
		//! Create a deep copy of the property
		virtual std::unique_ptr<PropertyBase> clone() const override
		{
			auto prop = std::make_unique<Property<T, index_type>>(name());

			prop->_allocPolicy = std::make_unique<Core::AlignedAllocPolicy<value_type, 32>>(*_allocPolicy);
			prop->_data = prop->_allocPolicy->allocate(size());
			prop->_allocated = size();
			prop->setSize(size());
			
			for (size_t i = 0; i < size(); i++)
				prop->_data[i] = _data[i];

			return std::move(prop);
		}
		
		virtual std::unique_ptr<PropertyBase> create() const override
		{
			auto prop = std::make_unique<Property<T, index_type>>(name());

			prop->_allocPolicy = std::make_unique<Core::AlignedAllocPolicy<value_type, 32>>(*_allocPolicy);

			return std::move(prop);
		}

		void assign(reference v)
		{
			for (int i = 0; i < size(); i++)
				_data[i] = v;
		}
		
		void assign(rvalue_reference v)
		{
			for (int i = 0; i < size(); i++)
				_data[i] = v;
		}

		template<typename AllocPolicyT>
		void setAllocator(AllocPolicyT& alloc_policy)
		{
			std::unique_ptr<Core::AlignedAllocPolicy<value_type, 32>> old_alloc_policy = std::move(_allocPolicy);
			_allocPolicy = std::make_unique<Core::AlignedAllocPolicy<value_type, 32>>(alloc_policy);

			if (size() > 0)
			{
				// Create a new buffer and copy the data
				pointer data = _allocPolicy->allocate(size());
				for (size_t i = 0; i < size(); i++)
					new (data + i) value_type(std::move(_data[i]));

				// Release the old buffer
				old_alloc_policy->deallocate(_data, _allocated);
				_data = data;
				_allocated = size();
			}
		}
	
	public:
		//! Access the i'th element. No range check is performed!
		inline reference operator[](size_t idx)
		{
			VclRequire(idx < static_cast<int>(size()), "Index in bounds.");
			return _data[idx];
		}

		//! Const access to the i'th element. No range check is performed!
		inline const_reference operator[](size_t idx) const
		{
			VclRequire(idx < static_cast<int>(size()), "Index in bounds.");
			return _data[idx];
		}

		//! Access the i'th element. No range check is performed!
		inline reference operator[](index_type idx)
		{
			VclRequire(size() <= _allocated, "Used size is smaller/equal to allocated size.");
			VclRequire(idx.id() < size(), "Index in bounds.");
			return _data[idx.id()];
		}

		//! Const access to the i'th element. No range check is performed!
		inline const_reference operator[](index_type idx) const
		{
			VclRequire(size() <= _allocated, "Used size is smaller/equal to allocated size.");
			VclRequire(idx.id() < size(), "Index in bounds.");
			return _data[idx.id()];
		}

	public:
		index_type push_back(const_reference data)
		{
			VclRequire(size() <= _allocated, "Used size is smaller/equal to allocated size.");

			// Check if there is some allocated space left
			if (size() < _allocated)
			{
				_data[size()] = data;
				setSize(size() + 1);
			}
			else
			{
				// Copy the element in case the reference points to an element in this property
				value_type copy = data;

				// Reserve at least a minimum of 2 elements, else take the current allocation size times 1.5
				reserve(std::max(static_cast<size_t>(_allocated*1.5), std::max(static_cast<size_t>(2), _allocated)));
				_data[size()] = std::move(copy);
				setSize(size() + 1);
			}

			return index_type(typename index_type::IdType(size()) - 1);
		}

	public:
		pointer data() const
		{
			return _data;
		}

		//! Clear the current content
		virtual void clear() override
		{
			size_t count = size();
			setSize(0);
			
			for (size_t i = 0; i < count; i++)
				_data[i].~value_type();
		}

		//! Resize the container.
		virtual void resize(size_t count) override
		{
			VclRequire(size() <= _allocated, "Used size is smaller/equal to allocated size.");

			if (count == size())
				return;

			if (count < size())
			{
				// Call the destructor for the left over data
				for (size_t i = count; i < size(); i++)
					_data[i].~value_type();

				setSize(count);
				return;
			}

			if (count <= _allocated)
			{
				// Initialise the additional data
				for (size_t i = size(); i < count; i++)
				{
					//_data[i] = _defaultValue;
					new (_data + i) value_type(_defaultValue);
				}
				
				setSize(count);
			}
			else
			{
				// Create a new buffer and move the data
				pointer data = _allocPolicy->allocate(count);
				for (size_t i = 0; i < size(); i++)
					new (data + i) value_type(std::move(_data[i]));
				
				// Initialise the additional data
				for (size_t i = size(); i < count; i++)
					new (data + i) value_type(_defaultValue);

				// Release the old buffer
				_allocPolicy->deallocate(_data, _allocated);
				_data = data;
				setSize(count);
				_allocated = count;
			}
		}
		
		// Reserve additional memory without resizing the property. 
		// This requires allocating a new block of new memory and copying the old elements.
		virtual void reserve(size_t count) override
		{
			VclRequire(size() <= _allocated, "Used size is smaller/equal to allocated size.");

			if (count > _allocated)
			{
				// Create a new buffer and move the data
				pointer data = _allocPolicy->allocate(count);
				for (size_t i = 0; i < size(); i++)
					new (data + i) value_type(std::move(_data[i]));

				// Release the old buffer
				_allocPolicy->deallocate(_data, _allocated);
				_data = data;
				_allocated = count;
			}
		}

	private:
		pointer _data;

	private: /* Default value */
		//typename std::aligned_storage<sizeof(typename value_type), std::alignment_of<int>::value>::type _defaultValue;
		value_type _defaultValue;

	private:
		std::unique_ptr<Core::AlignedAllocPolicy<value_type, 32>> _allocPolicy;
	};
	
	template<typename ValueT, typename IndexT>
	class PropertyPtr
	{
	public:
		typedef ValueT	value_type;
		typedef IndexT	index_type;
	public:
		PropertyPtr() : _property(nullptr) {}
		PropertyPtr(const PropertyPtr<value_type, index_type>& other) { _property = other._property; }
		PropertyPtr(PropertyPtr<value_type, index_type>&& other) { _property = other._property; other._property = nullptr; }
		PropertyPtr(Property<value_type, index_type>* p) : _property(p) {}

		PropertyPtr& operator= (Property<value_type, index_type>* p)
		{
			_property = p;
			return *this;
		}

	public:
		operator Property<value_type, index_type>* () const { return _property; }
		operator const Property<value_type, index_type>* () const { return _property; }

		Property<value_type, index_type>* ptr() const { return _property; }

	public:
		typename Property<value_type, index_type>::const_reference operator[](size_t idx) const
		{
			return _property->operator[](idx);
		}

		typename Property<value_type, index_type>::reference operator[](size_t idx)
		{
			return _property->operator[](idx);
		}

		typename Property<value_type, index_type>::const_reference operator[](index_type idx) const
		{
			return _property->operator[](idx);
		}

		typename Property<value_type, index_type>::reference operator[](index_type idx)
		{
			return _property->operator[](idx);
		}

		Property<value_type, index_type>* operator -> () const
		{
			return _property;
		}

		Property<value_type, index_type>& operator * () const
		{
			return *_property;
		}

		operator bool() const
		{
			return (_property != nullptr);
		}

	private:
		Property<value_type, index_type>* _property;
	};
	
	template<typename ValueT, typename IndexT>
	class ConstPropertyPtr
	{
	public:
		typedef ValueT	value_type;
		typedef IndexT	index_type;

	public:
		ConstPropertyPtr() : _property(nullptr) {}

		ConstPropertyPtr(const PropertyPtr<value_type, index_type>& other) { _property = other.ptr(); }
		ConstPropertyPtr(const ConstPropertyPtr<value_type, index_type>& other) { _property = other.ptr(); }

		ConstPropertyPtr(PropertyPtr<value_type, index_type>&& other) { _property = other.ptr(); other._property = nullptr; }
		ConstPropertyPtr(ConstPropertyPtr<value_type, index_type>&& other) { _property = other.ptr(); other._property = nullptr; }

		ConstPropertyPtr(const Property<value_type, index_type>* p) : _property(p) {}

		ConstPropertyPtr& operator= (Property<value_type, index_type>* p)
		{
			_property = p;
		}
	public:
		operator const Property<value_type, index_type>* () const { return _property; }
		
		const Property<value_type, index_type>* ptr() const { return _property; }

	public:
		typename Property<value_type, index_type>::const_reference operator[](int idx) const
		{
			return _property->operator[](idx);
		}

		typename Property<value_type, index_type>::const_reference operator[](index_type idx) const
		{
			return _property->operator[](idx);
		}

		const Property<value_type, index_type>* operator -> () const
		{
			return _property;
		}

		const Property<value_type, index_type>& operator * () const
		{
			return *_property;
		}

		operator bool() const
		{
			return (_property != nullptr);
		}

	private:
		const Property<value_type, index_type>* _property;
	};
}}
