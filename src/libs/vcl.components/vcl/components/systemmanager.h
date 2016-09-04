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
#include <unordered_map>

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/components/entitymanager.h>
#include <vcl/components/system.h>

namespace Vcl { namespace Components
{
	/*!
	 * \class SystemManager
	 * \brief Base for all systems
	 */
	class SystemManager
	{
	public:
		SystemManager(Core::ref_ptr<EntityManager> em);

		SystemManager(const SystemManager&) = delete;
		SystemManager& operator=(const SystemManager&) = delete;

	public:
		/*!
		 *	Add a System to the SystemManager.
		 *
		 *	\returs a pointer to the added system
		 */
		template <typename S>
		Core::ref_ptr<S> add(Core::owner_ptr<S> system)
		{
			_systems.emplace_back(std::move(system));
			return _systems.back();
		}

		/*!
		 *	Add a System to the SystemManager.
		 *
		 *	\returs a pointer to the added system
		 */
		template <typename S, typename ... Args>
		Core::ref_ptr<S> add(Args && ... args)
		{			
			return add(Core::make_owner<S>(std::forward<Args>(args) ...));
		}

		/*!
		 *	Access the a registered System of an explicit instance.
		 *
		 *	\returns the pointer to a system of the requested type.
		 */
		template <typename S>
		Core::ref_ptr<S> system()
		{
			for (auto& sys : _systems)
			{
				if (dynamic_cast<S*>(sys.get()))
				{
					return sys;
				}
			}

			return{};
		}

	private:
		//! Entity manager these systems refer to
		Core::ref_ptr<EntityManager> _entities;

		//! Systems updated by this manager
		std::vector<Core::owner_ptr<System>> _systems;
	};
}}
