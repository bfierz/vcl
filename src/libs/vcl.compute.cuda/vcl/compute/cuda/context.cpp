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
#include <vcl/compute/cuda/context.h>

// C++ standard library

// CUDA GL support
#ifdef VCL_OPENGL_SUPPORT
#	include <GL/glew.h>
#	include <cudaGL.h>
#endif // VCL_OPENGL_SUPPORT

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/commandqueue.h>
#include <vcl/compute/cuda/module.h>
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace Cuda
{
	Context::Context(const Device& dev, ApiBinding binding)
	: Compute::Context()
	, _dev(dev)
	{
		// Device ID
		CUdevice dev_id = dev;

		// Initialize the new context
		unsigned int flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;

		switch (binding)
		{
		case ApiBinding::None:
			break;

		case ApiBinding::OpenGL:
		{
#ifdef VCL_OPENGL_SUPPORT

			int deviceCount = 0;
			VCL_CU_SAFE_CALL(cuDeviceGetCount(&deviceCount));

			if (deviceCount > 0)
			{
				unsigned int foundDevices = 0;
				std::vector<CUdevice> compatibleDevices(deviceCount);

				VCL_CU_SAFE_CALL(cuGLGetDevices(&foundDevices, compatibleDevices.data(), deviceCount, CU_GL_DEVICE_LIST_ALL));

				if (foundDevices > 0)
				{
					auto result = std::find(compatibleDevices.begin(), compatibleDevices.end(), dev_id);
					Check(result != compatibleDevices.end(), "Device is OpenGL compatible.");

					VCL_CU_SAFE_CALL(cuGLCtxCreate(&_context, flags, dev_id));
				}
			}
#else
			DebugError("OpenGL support not compiled into the library.");
#endif // VCL_OPENGL_SUPPORT
			break;
		}
		default:
			DebugError("Unkown API binding.");
		};

		// Default creation
		if (!_context)
		{
			VCL_CU_SAFE_CALL(cuCtxCreate(&_context, flags, dev_id));
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
			VCL_CU_SAFE_CALL(cuCtxDestroy(_context));
			_context = nullptr;
		}
	}

	bool Context::isCurrent() const
	{
		CUcontext ctx;
		VCL_CU_SAFE_CALL(cuCtxGetCurrent(&ctx));

		return ctx == _context;
	}

	void Context::bind()
	{
		VCL_CU_SAFE_CALL(cuCtxSetCurrent(this->operator CUcontext()));
	}

	void Context::sync()
	{
		VCL_CU_SAFE_CALL(cuCtxSynchronize());
	}

	Context::ref_ptr<Compute::Module> Context::createModuleFromSource(const int8_t* source, size_t size)
	{
		_modules.emplace_back(Module::loadFromBinary(this, source, size));
		return _modules.back();
	}
	Context::ref_ptr<Compute::Buffer> Context::createBuffer(BufferAccess access, size_t size)
	{
		_buffers.emplace_back(Core::make_owner<Cuda::Buffer>(this, access, size));
		return _buffers.back();
	}
	Context::ref_ptr<Compute::CommandQueue> Context::createCommandQueue()
	{
		_queues.emplace_back(Core::make_owner<Cuda::CommandQueue>(this));
		return _queues.back();
	}
}}}
