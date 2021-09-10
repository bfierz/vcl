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
#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

// VCL
#include <vcl/components/entity.h>
#include <vcl/rtti/metatypelookup.h>

namespace Vcl { namespace Components {
	template<typename T>
	struct ComponentTraits
	{
		static const bool IsUnique{ true };
	};

	/*!
	 *	\class ComponentStoreBase
	 *	\brief Base class for component storage
	 */
	class ComponentStoreBase
	{
	public:
		ComponentStoreBase(const Vcl::RTTI::Type* type)
		: _type(type) {}
		virtual ~ComponentStoreBase() = default;

		//! \returns the type of the components in this store
		const Vcl::RTTI::Type* type() const { return _type; }

		//! \returns true if the store is empty
		virtual bool empty() const = 0;

		//! \returns true if the entity as a component associated in this store
		virtual bool has(EntityId id) const = 0;

		virtual void store(EntityId id, std::vector<std::pair<void*, const Vcl::RTTI::Type*>>& components) const = 0;

	private:
		//! The type of the components in this store
		const Vcl::RTTI::Type* _type;
	};

	/*!
	 *	\class ComponentStore
	 *	\brief Manage the live time of a single component type
	 *
	 *	This class is the heart of the component system. It efficiently stores
	 *	and finds components of an entity.
	 */
	template<typename T>
	class ComponentStore : public ComponentStoreBase
	{
	public:
		using ComponentType = T;

	public:
		ComponentStore()
		: ComponentStoreBase(vcl_meta_type<ComponentType>()) {}

		bool empty() const override
		{
			return _components.empty();
		}

		bool has(EntityId id) const override
		{
			return _components.find(id) != _components.end();
		}

		auto operator()(EntityId id) -> ComponentType*
		{
			return &_components.find(id)->second;
		}

		void store(EntityId id, std::vector<std::pair<void*, const Vcl::RTTI::Type*>>& components) const override
		{
			const auto comp = _components.find(id);
			if (comp != _components.end())
			{
				components.emplace_back((void*)&comp->second, type());
			}
		}

		auto operator()(EntityId id) const -> const ComponentType*
		{
			return &_components.find(id)->second;
		}

		template<typename Func>
		void forEach(Func&& f) const
		{
			for (auto& entry : _components)
			{
				f(entry.first, &entry.second);
			}
		}

	public:
		template<typename... Args>
		auto create(EntityId id, Args... args) -> ComponentType*
		{
			VclRequire(!has(id), "Component for entity does not exist.");

			auto newElemIter = _components.emplace(id, std::forward<Args>(args)...);

			VclEnsure(newElemIter.second, "Element was inserted.");
			return &newElemIter.first->second;
		}

	public:
		std::unordered_map<EntityId, ComponentType> _components;
	};

	template<typename T>
	class MultiComponentStoreBase : public ComponentStoreBase
	{
	public:
		using ComponentType = T;
		using Store = std::unordered_multimap<EntityId, ComponentType>;

	public:
		MultiComponentStoreBase()
		: ComponentStoreBase(vcl_meta_type<ComponentType>()) {}

		bool empty() const override
		{
			return _components.empty();
		}

		bool has(EntityId id) const override
		{
			return _components.find(id) != _components.end();
		}

		void store(EntityId id, std::vector<std::pair<void*, const Vcl::RTTI::Type*>>& components) const override
		{
			auto itr = (*this)(id);
			for (auto it = itr.first; it != itr.second; ++it)
			{
				components.emplace_back((void*)&it->second, type());
			}
		}

		auto operator()(EntityId id) -> std::pair<typename Store::iterator, typename Store::iterator>
		{
			return _components.equal_range(id);
		}

		auto operator()(EntityId id) const -> std::pair<typename Store::const_iterator, typename Store::const_iterator>
		{
			return _components.equal_range(id);
		}

	public:
		template<typename... Args>
		auto create(EntityId id, Args... args) -> ComponentType*
		{
			auto newElemIter = _components.emplace(id, std::forward<Args>(args)...);

			VclEnsure(newElemIter != _components.end(), "Element was inserted.");
			return &newElemIter->second;
		}

	protected:
		//! Storage of the allocated components
		Store _components;
	};

	/*!
	 *	\class MultiComponentStore
	 *	\brief Manage the live time of a single component type
	 *
	 *	This class is the heart of the component system. It efficiently stores
	 *	and finds components of an entity. In contrast to many other implementations
	 *	this class allows to store several components of the same type for an entity.
	 */
	template<typename T, typename Func>
	class MultiComponentStore : public MultiComponentStoreBase<T>
	{
	public:
		using ComponentType = T;

	public:
		MultiComponentStore(Func&& f)
		: _func(std::forward<Func&&>(f))
		{
		}

	public:
		template<typename IdType>
		auto operator()(EntityId id, const IdType&& comp_id) -> ComponentType*
		{
			// For some reason the 'this->' is important for GCC
			auto components = this->_components.equal_range(id);
			auto comp_iter = std::find_if(components.first, components.second, std::bind(_func, comp_id));

			if (comp_iter != components.end())
				return comp_iter->second;
			else
				return nullptr;
		}

	private:
		//! Functor to identify a single component for an entity
		Func _func;
	};
}}
