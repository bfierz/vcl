#pragma once

// VCL library configuration
#include <vcl/config/global.h>

// C++ standard library
#include <numeric_limits>

// VCL

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
		GenericId(const GenericId<Derived, T>& rhs) : _id(rhs._id) {}

	public:
		explicit T id() const { return _id; }
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
}}

// Instantiate a generic, typed ID

#define VCL_CREATEID(type_name, idx_type_name) class type_name : public Vcl::Core::GenericId<type_name, idx_type_name> { public: type_name(){} explicit type_name(idx_type_name id) : GenericId<type_name, idx_type_name>(id) {}}


namespace Vcl { namespace Component
{
	/*!
	 * \class Entity
	 * \brief Representation of a single entity
	 */
	VCL_CREATEID(Entity, uint32_t);
	
	/*!
	 * \class EntityManager
	 * \brief Create and manage the live time of all entities
	 */
	class EntityManager
	{
	
	public:
		Entity create();
		
		bool isAlive();
	};
}}
