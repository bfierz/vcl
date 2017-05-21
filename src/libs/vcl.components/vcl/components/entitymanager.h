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
#include <vcl/components/componentstore.h>
#include <vcl/components/entity.h>

namespace Vcl { namespace Components
{
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
		template<typename C>
		void registerComponent()
		{
			VclRequire(_components.find(typeid(C).hash_code()) == _components.end(), "Component type not registered.");

			size_t hash = typeid(C).hash_code();
			if (_components.find(hash) == _components.end())
			{
				_components[hash] = std::make_unique<ComponentStore<C>>();
			}

			VclEnsure(_components.find(typeid(C).hash_code()) != _components.end(), "Component type registered.");
		}

		template<typename C, typename Func>
		void registerComponent(Func&& func)
		{
			VclRequire(_multiComponents.find(typeid(C).hash_code()) == _multiComponents.end(), "Component type not registered.");

			size_t hash = typeid(C).hash_code();
			if (_multiComponents.find(hash) == _multiComponents.end())
			{
				_multiComponents[hash] = std::make_unique<MultiComponentStore<C, Func>>(std::forward<Func&&>(func));
			}

			VclEnsure(_multiComponents.find(typeid(C).hash_code()) != _multiComponents.end(), "Component type registered.");
		}

		template<typename C, typename... Args>
		typename std::enable_if<ComponentTraits<C>::IsUnique, C*>::type
			create(Entity e, Args&&... args)
		{
			VclRequire(_components.find(typeid(C).hash_code()) != _components.end(), "Component type registered.");
			VclRequire(e._manager == this, "Entity belongs the this system.");

			size_t hash = typeid(C).hash_code();
			auto c = static_cast<ComponentStore<C>*>(_components[hash].get());

			return c->create(e._id, std::forward<Args>(args)...);
		}

		template<typename C, typename... Args>
		typename std::enable_if<!ComponentTraits<C>::IsUnique, C*>::type
			create(Entity e, Args&&... args)
		{
			VclRequire(_multiComponents.find(typeid(C).hash_code()) != _multiComponents.end(), "Component type registered.");
			VclRequire(e._manager == this, "Entity belongs the this system.");

			size_t hash = typeid(C).hash_code();
			auto c = static_cast<MultiComponentStoreBase<C>*>(_multiComponents[hash].get());
			
			return c->create(e._id, std::forward<Args>(args)...);
		}

		template<typename C>
		const typename std::enable_if<ComponentTraits<C>::IsUnique, ComponentStore<C>>::type*
			get() const
		{
			VclRequire(_components.find(typeid(C).hash_code()) != _components.end(), "Component type registered.");

			size_t hash = typeid(C).hash_code();
			auto c = static_cast<const ComponentStore<C>*>(_components.find(hash)->second.get());

			return c;
		}

		template<typename C>
		const typename std::enable_if<!ComponentTraits<C>::IsUnique, MultiComponentStoreBase<C>>::type*
			get() const
		{
			VclRequire(_multiComponents.find(typeid(C).hash_code()) != _multiComponents.end(), "Component type registered.");

			size_t hash = typeid(C).hash_code();
			auto c = static_cast<const MultiComponentStoreBase<C>*>(_multiComponents.find(hash)->second.get());

			return c;
		}

		template<typename C>
		bool has(Entity e)
		{
			VclRequire(e._manager == this, "Entity belongs the this system.");

			size_t hash = typeid(C).hash_code();
			if (ComponentTraits<C>::IsUnique)
			{
				auto& c = *static_cast<ComponentStore<C>*>(_components.find(hash)->second.get());

				return c.has(e._id);
			}
			else
			{
				auto& c = *static_cast<MultiComponentStoreBase<C>*>(_multiComponents.find(hash)->second.get());

				return c.has(e._id);
			}
		}

	public:
		//! \returns the number of active entities
		size_t size() const
		{
			return _generations.size() - _freeIndices.size();
		}

		//! \returns the current capacity of the manager
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

		//! Per type storage of unique components per entity
		std::unordered_map<size_t, std::unique_ptr<ComponentStoreBase>> _components;

		//! Per type stortage of multi-components per entity
		std::unordered_map<size_t, std::unique_ptr<ComponentStoreBase>> _multiComponents;
	};
}}
