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
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

// VCL
#include <vcl/components/entity.h>

namespace Vcl { namespace Components
{
	/*!
	 *	\class ComponentStoreBase
	 *	\brief Base class for component storage
	 */
	class ComponentStoreBase
	{

	};

	/*!
	 *	\class ComponentStore
	 *	\brief Manage the live time of a single component type
	 *
	 *	This class is the heart of the component system. It efficiently stores
	 *	and finds components of an entity. In contrast to many other implementations
	 *	this class allows to store several components of the same type for an entity.
	 */
	template<typename T>
	class ComponentStore : public ComponentStoreBase
	{
	public:
		using ComponentType = T;

	public:
		T* operator[](EntityId id)
		{
			if (id.id() >= _components.size())
			{
				_components.resize(id.id() + 1);
				_valid.resize(id.id() + 1);
			}

			return &_components[id.id()];
		}

		void setValid(EntityId id, bool b)
		{
			_valid[id.id()] = b;
		}

		bool isValid(EntityId id)
		{
			return _valid[id.id()];
		}

	public:
		std::vector<ComponentType> _components;
		std::vector<bool> _valid;
	};
	
	/*!
	 *	\class ComponentPtr
	 *	\brief Pointer to a single component entry
	 */
	template<typename T>
	class ComponentPtr
	{
	public:
		using ComponentType = T;

	public:
		ComponentPtr(ComponentStore<ComponentType>& store, EntityId idx)
		: _store(store)
		, _idx(idx)
		{}

	public:
		ComponentType* operator->() const
		{
			return _store[_idx];
		}

	private:
		ComponentStore<ComponentType>& _store;

		EntityId _idx;
	};


	/*!
	 *	\class EntityManager
	 *	\brief Create and manage the live time of all entities
	 *
	 *	\note Inspiration was gathered through various articles and http://entity-systems.wikidot.com/
	 */
	class EntityManager
	{
	public:
		/*!
		 *	\brief Creates a new entity
		 *	\returns the link to a new entity
		 */
		Entity create();

		/*!
		 *	\brief Destroys an entity
		 */
		void destroy(Entity e);

	public:
		template<typename C, typename... Args>
		ComponentPtr<C> create(Entity e, Args&&... args)
		{
			Require(e._manager == this, "Entity belongs the this system.");

			size_t hash = typeid(C).hash_code();
			if (_components.find(hash) == _components.end())
			{
				_components[hash] = std::make_unique<ComponentStore<C>>();
			}

			auto& c = *static_cast<ComponentStore<C>*>(_components[hash].get());
			new(c[e._id]) C(std::forward<Args>(args) ...);
			c.setValid(e._id, true);

			return{ c, e._id };
		}

		template<typename C>
		bool has(Entity e)
		{
			Require(e._manager == this, "Entity belongs the this system.");

			size_t hash = typeid(C).hash_code();
			auto& c = *static_cast<ComponentStore<C>*>(_components[hash].get());
			
			return c.isValid(e._id);
		}

	public:
		//! @returns the number of active entities
		size_t size() const
		{
			return _generations.size() - _freeIndices.size();
		}

		//! @returns the current capacity of the manager
		size_t capacity() const
		{
			return _generations.size();
		}

	private: // Entity management
		//! List of allocated entity-entries. Its size gives the number of in total allocated entities.
		std::vector<uint32_t> _generations;

		//! List of released entities
		std::vector<uint32_t> _freeIndices;

	private: // Component store
		std::unordered_map<size_t, std::unique_ptr<ComponentStoreBase>> _components;

	};
}}
