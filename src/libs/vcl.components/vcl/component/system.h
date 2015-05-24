#pragma once

// VCL library configuration
#include <vcl/config/global.h>

// C++ standard library
#include <string>

// VCL
#include <vcl/component/entitymanager.h>

namespace Vcl { namespace Component
{
	/*!
	 * \class System
	 * \brief Base for all systems
	 */
	class System
	{
	public:
		System(SystemManager* owner);
	
	public:
		bool hasComponent(Entity e) const;

	private:
		SystemManager* _owner;
		
		//! Readable name of the system
		std::string _name;
	};

	/*!
	 * \class EntityNameSystem
	 * \brief Simple system assigning each entity a name
	 */
	class EntityNameSystem : public System
	{
	public:
		void setName(Entity e, const std::string& name);
		const std::string& name(Entity e) const;
		
	private:
		//! Map of entities to their names
		std::unordered_map<Entity, std::string> _entityNames;
	};
	
	/*!
	 * \class TransformSystem
	 * \brief Simple system assigning each entity a spatial transformation
	 */
	class TransformSystem : public System
	{
	public:
		void setRotation(Entity e, const Quaternionf& rot);
		const Quaternionf& rotation(Entity e) const;
		
		void setTranslation(Entity e, const Vector3f& rot);
		const Vector3f& translation(Entity e) const;
		
	private:
	};
}}
