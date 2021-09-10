/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#include <string>
#include <vector>

// VCL
#include <vcl/compute/commandqueue.h>
#include <vcl/compute/buffer.h>
#include <vcl/compute/module.h>
#include <vcl/core/memory/smart_ptr.h>

namespace Vcl { namespace Compute {
	class Context
	{
	public:
		template<typename T>
		using owner_ptr = Vcl::Core::owner_ptr<T>;
		template<typename T>
		using ref_ptr = Vcl::Core::ref_ptr<T>;

	public:
		//! Constructor
		Context() = default;

		//! Context is not copyable
		Context(const Context&) = delete;
		Context& operator=(const Context&) = delete;

		//! Destructor
		virtual ~Context() = default;

		//! Access the default command queue
		ref_ptr<CommandQueue> defaultQueue() const;

	public: // Resource allocation
		virtual ref_ptr<Module> createModuleFromSource(const int8_t* source, size_t size) = 0;

		virtual ref_ptr<Buffer> createBuffer(BufferAccess access, size_t size) = 0;

		virtual ref_ptr<CommandQueue> createCommandQueue() = 0;

		void release(ref_ptr<Module> h);
		void release(ref_ptr<Buffer> h);
		void release(ref_ptr<CommandQueue> h);

	protected: // Resources
		//! All allocated buffers on this device
		std::vector<owner_ptr<Buffer>> _buffers;

		//! All allocated modules on this device
		std::vector<owner_ptr<Module>> _modules;

		//! All allocated streams on this device
		std::vector<owner_ptr<CommandQueue>> _queues;
	};
}}
