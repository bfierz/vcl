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
#include <vcl/compute/opencl/context.h>

// C++ standard library

// VCL
#include <vcl/compute/opencl/buffer.h>
#include <vcl/compute/opencl/commandqueue.h>
#include <vcl/compute/opencl/module.h>
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace OpenCL {
	Context::Context(const Device& dev)
	: Compute::Context()
	, _dev(dev)
	{
		// Device ID
		cl_device_id dev_id = dev;

		// Get the platform associated with the device
		cl_platform_id platform;
		VCL_CL_SAFE_CALL(clGetDeviceInfo(dev_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, nullptr));

		// Create context without graphics bindings
		{
			cl_context_properties props[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties)platform, // OpenCL platform
				0
			};

			cl_int err;
			_context = clCreateContext(props, 1, &dev_id, nullptr, nullptr, &err);
			VCL_CL_SAFE_CALL(err);
		}

		// Create the main command stream
		if (_context)
		{
			createCommandQueue();
		}
	}

	Context::~Context()
	{
		if (_context)
		{
			VCL_CL_SAFE_CALL(clReleaseContext(_context));
			_context = nullptr;
		}
	}

	Context::ref_ptr<Compute::Module> Context::createModuleFromSource(const int8_t* source, size_t size)
	{
		_modules.emplace_back(Module::loadFromSource(this, source, size));
		return _modules.back();
	}
	Context::ref_ptr<Compute::Buffer> Context::createBuffer(BufferAccess access, size_t size)
	{
		_buffers.emplace_back(Core::make_owner<OpenCL::Buffer>(this, access, size));
		return _buffers.back();
	}
	Context::ref_ptr<Compute::CommandQueue> Context::createCommandQueue()
	{
		_queues.emplace_back(Core::make_owner<OpenCL::CommandQueue>(this));
		return _queues.back();
	}
}}}
