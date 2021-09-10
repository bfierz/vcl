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

// VCL
#include <vcl/compute/context.h>

#include <vcl/compute/cuda/device.h>

namespace Vcl { namespace Compute { namespace Cuda {
	enum class ApiBinding
	{
		None,
		OpenGL,
		Direct3D11
	};

	class Context : public Compute::Context
	{
	public:
		//! Constructor
		Context(const Device&, ApiBinding binding = ApiBinding::None);

		//! Destructor
		virtual ~Context();

		//! Convert to OpenCL device ID
		inline operator CUcontext() const
		{
			return _context;
		}

		//! Bind the context to the current thread
		void bind();

		//! Wait for all the operations on this context to finish
		void sync();

	public: // Resource allocation
		virtual ref_ptr<Compute::Module> createModuleFromSource(const int8_t* source, size_t size) override;
		virtual ref_ptr<Compute::Buffer> createBuffer(BufferAccess access, size_t size) override;
		virtual ref_ptr<Compute::CommandQueue> createCommandQueue() override;

	public:
		const Device& device() const { return _dev; }

	public:
		bool isCurrent() const;

	private:
		//! OpenCL context ID
		CUcontext _context{ nullptr };

		//! Device belonging to this context
		const Device& _dev;
	};
}}}
