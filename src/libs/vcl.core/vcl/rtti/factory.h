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

// VCL
#include <vcl/core/rtti/metatype.h>
#include <vcl/core/rtti/metatyperegistry.h>

namespace Vcl { namespace RTTI
{
	class Factory
	{
	public:
		template<typename... Args>
		static void* create(const char* name, Args... args)
		{
			// Get the meta type
			auto type = vcl_meta_type(name);

			// Type is available
			if (!type)
				return nullptr;

			if (type->_isConstructable)
			{
				return create(name, boost::any(args)...);
			}

			// Allocate memory for a new type instance
			auto inst = type->allocate();

			// Call the default constructor
			type->construct(inst, args...);

			return inst;
		}

		static void* create(const char* name)
		{
			// Get the meta type
			auto type = vcl_meta_type(name);

			// Is type constructable
			if (!type || type->_isConstructable == false)
				return nullptr;

			// Allocate memory for a new type instance
			auto inst = type->allocate();
			
			// Call the default constructor
			std::initializer_list<boost::any> params = {};
			type->construct(inst, params);

			return inst;
		}
		
		static void* create(const char* name, const boost::any& v0)
		{
			// Get the meta type
			auto type = vcl_meta_type(name);

			// Is type constructable
			if (!type || type->_isConstructable == false)
				return nullptr;

			// Allocate memory for a new type instance
			auto inst = type->allocate();
			
			// Call the default constructor
			std::initializer_list<boost::any> params = {v0};
			type->construct(inst, params);

			return inst;
		}
		
		static void* create(const char* name, const boost::any& v0, const boost::any& v1)
		{
			// Get the meta type
			auto type = vcl_meta_type(name);

			// Is type constructable
			if (!type || type->_isConstructable == false)
				return nullptr;

			// Allocate memory for a new type instance
			auto inst = type->allocate();
			
			// Call the default constructor
			std::initializer_list<boost::any> params = {v0, v1};
			type->construct(inst, params);

			return inst;
		}
	};
}}
