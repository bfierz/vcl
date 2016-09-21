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

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Core
{
	template<typename Derived, typename T = unsigned int>
	class GenericId
	{
	public:
		typedef T IdType;

	public:
		static Derived InvalidId() { return Derived(std::numeric_limits<T>::max()); }

	public:
		GenericId() : _id(InvalidId().id()) {}
		explicit GenericId(T id) : _id(id) {}

	public:
		T id() const { return _id; }
		bool isValid() const { return _id != InvalidId().id(); }

	public:
		GenericId<Derived, T>& operator = (const GenericId<Derived, T>& rhs)
		{
			_id = rhs._id;
			return *this;
		}

	public:
		bool operator < (const GenericId<Derived, T>& rhs) const
		{
			return _id < rhs._id;
		}

		bool operator <= (const GenericId<Derived, T>& rhs) const
		{
			return _id <= rhs._id;
		}

		bool operator > (const GenericId<Derived, T>& rhs) const
		{
			return _id > rhs._id;
		}

		bool operator >= (const GenericId<Derived, T>& rhs) const
		{
			return _id >= rhs._id;
		}

		bool operator == (const GenericId<Derived, T>& rhs) const
		{
			return _id == rhs._id;
		}

		bool operator != (const GenericId<Derived, T>& rhs) const
		{
			return _id != rhs._id;
		}

	protected:
		T _id;
	};

	template<typename Derived, typename T = unsigned int, typename U = unsigned int>
	class GenerationalId
	{
	public:
		typedef T IdType;
		typedef U GenerationType;

	public:
		static Derived InvalidId() { return Derived(std::numeric_limits<IdType>::max(), std::numeric_limits<GenerationType>::max()); }

	public:
		GenerationalId()
		: _id(InvalidId().id())
		, _generation(InvalidId().generation())
		{}
		explicit GenerationalId(IdType id, GenerationType generation)
		: _id(id)
		, _generation(generation)
		{}

	public:
		IdType id() const { return _id; }
		GenerationType generation() const { return _generation; }
		bool isValid() const { return _id != InvalidId().id() && _generation != InvalidId().generation(); }

	public:
		bool operator < (const GenerationalId<Derived, IdType, GenerationType>& rhs) const
		{
			return _generation < rhs._generation && _id < rhs._id;
		}

		bool operator <= (const GenerationalId<Derived, IdType, GenerationType>& rhs) const
		{
			return _generation <= rhs._generation && _id <= rhs._id;
		}

		bool operator >(const GenerationalId<Derived, IdType, GenerationType>& rhs) const
		{
			return _generation > rhs._generation && _id > rhs._id;
		}

		bool operator >= (const GenerationalId<Derived, IdType, GenerationType>& rhs) const
		{
			return _generation >= rhs._generation && _id >= rhs._id;
		}

		bool operator == (const GenerationalId<Derived, IdType, GenerationType>& rhs) const
		{
			return _generation == rhs._generation && _id == rhs._id;
		}

		bool operator != (const GenerationalId<Derived, IdType, GenerationType>& rhs) const
		{
			return _generation != rhs._generation && _id != rhs._id;
		}

	protected:
		IdType _id;
		GenerationType _generation;
	};
}}

// Instantiate a generic, typed ID
#define VCL_CREATE_ID(type_name, idx_type_name) class type_name : public Vcl::Core::GenericId<type_name, idx_type_name> { public: type_name() = default; explicit type_name(idx_type_name id) : GenericId<type_name, idx_type_name>(id) {}}
#define VCL_CREATE_GENERATIONALID(type_name, idx_type_name, gen_type_name) class type_name : public Vcl::Core::GenerationalId<type_name, idx_type_name, gen_type_name> { public: type_name() = default; explicit type_name(idx_type_name id, gen_type_name gen) : GenerationalId(id, gen) {}}


namespace Vcl { namespace Components
{
	// Forward declaration
	class EntityManager;

	/*!
	 * \class Entity
	 * \brief Representation of a single entity
	 */
	VCL_CREATE_GENERATIONALID(EntityId, uint32_t, uint32_t);

	/*!
	 * \class Entity
	 * \brief Representation of an entity
	 */
	class Entity
	{
		friend class EntityManager;
	public:
		Entity() = default;
		~Entity() = default;

	private:
		Entity(EntityManager* em, uint32_t idx, uint32_t gen)
		: _manager(em)
		, _id(idx, gen)
		{
			Ensure(_manager, "Entity manager is set.");
			Ensure(_id.isValid(), "Id is valid.");
		}

	private:
		//! Link to the manager this entity belongs to
		EntityManager* _manager{ nullptr };

		//! Id of the Entity
		EntityId _id;
	};
}}

namespace std
{
	template<>
	struct hash<Vcl::Components::EntityId>
	{
		size_t operator()(const Vcl::Components::EntityId& id) const
		{
			size_t h1 = std::hash<uint32_t>()(id.id());
			size_t h2 = std::hash<uint32_t>()(id.generation());
			return h1 ^ (h2 << 1);
		}
	};
}
