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
#include <vcl/compute/cuda/platform.h>

// C++ standard library
#include <array>
#include <iostream>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace Cuda
{
	Platform* Platform::_implementation = nullptr;

	void Platform::initialise()
	{
		if (_implementation == nullptr)
			_implementation = new Platform;
	}

	Platform* Platform::instance()
	{
		VclCheck(_implementation != nullptr, "CUDA platorm is initialised.");
		return _implementation;
	}

	void Platform::dispose()
	{
		VCL_SAFE_DELETE(_implementation);
	}

	Platform::Platform()
	{
		// Initialize CUDA
		VCL_CU_SAFE_CALL(cuInit(0));

		// Get the number of CUDA devices
		int nr_devices;
		VCL_CU_SAFE_CALL(cuDeviceGetCount(&nr_devices));
		_devices.reserve(nr_devices);

		// Load the CUDA devices
		for (int i = 0; i < nr_devices; i++)
		{
			CUdevice dev;
			VCL_CU_SAFE_CALL(cuDeviceGet(&dev, i));
			_devices.emplace_back(dev);
		}
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
		VclRequire(idx < _devices.size(), "idx is valid.");

		return _devices[idx];
	}

	int Platform::version() const
	{
		int version;
		VCL_CU_SAFE_CALL(cuDriverGetVersion(&version));
		return version;
	}
}}}
