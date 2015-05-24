#pragma once

// VCL library configuration
#include <vcl/config/global.h>

// C++ standard library
#include <vector>

// VCL
#include <vcl/component/system.h>

namespace Vcl { namespace Component
{
	/*!
	 * \class SystemManager
	 * \brief Base for all systems
	 */
	class SystemManager
	{
	public:
		SystemManager();
	
	public:

	private:
		std::vector<System> _systems;
	};
}}
