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
#include <vcl/compute/opencl/platform.h>

// C++ standard library
#include <cstring>
#include <iostream>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace OpenCL
{
	Platform* Platform::_implementation = nullptr;

	void Platform::initialise()
	{
		if (_implementation == nullptr)
			_implementation = new Platform;
	}

	Platform* Platform::instance()
	{
		Check(_implementation != nullptr, "OpenCL platorm is initialised.");
		return _implementation;
	}

	void Platform::dispose()
	{
		VCL_SAFE_DELETE(_implementation);
	}

	Platform::Platform()
	{
		cl_platform_id* platforms = nullptr;
		char buffer[256] = {0};
		unsigned int buffer_size = 256 * sizeof(char);
		cl_uint nr_platforms = 0;

		// Get the number of available platforms
		VCL_CL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &nr_platforms));

		// Allocate space for platform IDs
		platforms = new cl_platform_id[nr_platforms];
		if (platforms == nullptr)
		{
			DebugError("TODO.");
		}

		// Get the available platforms
		VCL_CL_SAFE_CALL(clGetPlatformIDs(nr_platforms, platforms, nullptr));

		for (cl_uint ui = 0; ui < nr_platforms; ui++)
		{
			cl_int err;
			err = clGetPlatformInfo(platforms[ui], CL_PLATFORM_NAME, buffer_size, buffer, NULL);
			if (err == CL_SUCCESS)
			{
				std::cout << ui << ": OpenCL name: " << buffer << std::endl;
			}
			
			err = clGetPlatformInfo(platforms[ui], CL_PLATFORM_PROFILE, buffer_size, buffer, NULL);
			if (err == CL_SUCCESS)
			{
				std::cout << ui << ": OpenCL profile: " << buffer << std::endl;
			}
			
			err = clGetPlatformInfo(platforms[ui], CL_PLATFORM_VERSION, buffer_size, buffer, NULL);
			if (err == CL_SUCCESS)
			{
				std::cout << ui << ": OpenCL version: " << buffer << std::endl;
			}

			err = clGetPlatformInfo(platforms[ui], CL_PLATFORM_VENDOR, buffer_size, buffer, NULL);
			if (err == CL_SUCCESS)
			{
				std::cout << ui << ": OpenCL vendor: " << buffer << std::endl;
			}

			size_t extension_buffer_size = 0;
			err = clGetPlatformInfo(platforms[ui], CL_PLATFORM_EXTENSIONS, 0, NULL, &extension_buffer_size);
			if (err == CL_SUCCESS)
			{
				std::vector<char> extension_buffer(extension_buffer_size, 0);
				err = clGetPlatformInfo(platforms[ui], CL_PLATFORM_EXTENSIONS, extension_buffer_size, extension_buffer.data(), NULL);
				if (err == CL_SUCCESS)
				{
					std::cout << ui << ": OpenCL extensions:" << std::endl;

					std::string str_buffer = extension_buffer.data();
					std::string::size_type head = 0;
					std::string::size_type tail = str_buffer.find(' ');

					while (tail != str_buffer.npos)
					{
						std::cout << "\t" << str_buffer.substr(head, tail) << std::endl;

						head = tail + 1;
						tail = str_buffer.find(' ', head);
					}
				}
			}
			else
			{
				std::cout << "OpenCL error: " << err << std::endl;
			}
		}
		
		// Query devices
		for (cl_uint ui = 0; ui < nr_platforms; ui++)
		{
			cl_int err;
			cl_uint nr_devices;
			err = clGetDeviceIDs(platforms[ui], CL_DEVICE_TYPE_ALL, 0, nullptr, &nr_devices);
			if (err == CL_SUCCESS)
			{
				std::vector<cl_device_id> devices(nr_devices, nullptr);
				err = clGetDeviceIDs(platforms[ui], CL_DEVICE_TYPE_ALL, nr_devices, devices.data(), 0);
				if (err == CL_SUCCESS)
				{
					for (auto dev : devices)
						_devices.emplace_back(dev);
				}
			}
		}

		VCL_SAFE_DELETE(platforms);

	}

	Platform::~Platform()
	{
		_devices.clear();
	}

	int Platform::nrDevices() const
	{
		return static_cast<int>(_devices.size());
	}

	const Device& Platform::device(int idx) const
	{
		Require(idx < _devices.size(), "idx is valid.");

		return _devices[idx];
	}

	int Platform::version() const
	{
		int version;
		//VCL_CU_SAFE_CALL(cuDriverGetVersion(&version));
		return version;
	}
}}}
