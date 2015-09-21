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
#include <vcl/compute/cuda/device.h>

// C++ standard library
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace Cuda
{
	Device::Device(CUdevice dev)
	: _device(dev)
	{
		// Read the device name
		std::vector<char> buffer(1024, 0);
		VCL_CU_SAFE_CALL(cuDeviceGetName(buffer.data(), buffer.size(), _device));
		_name = buffer.data();

		// Read the compute capability
		int capMajor, capMinor;
		VCL_CU_SAFE_CALL(cuDeviceComputeCapability(&capMajor, &capMinor, _device));

		if (capMajor >= 5)
		{
			_capability = DeviceCapability::Sm50;
		}
		else if (capMajor == 3)
		{
			if (capMinor >= 5)
				_capability = DeviceCapability::Sm35;
			else
				_capability = DeviceCapability::Sm30;
		}
		else if (capMajor == 2)
		{
			_capability = DeviceCapability::Sm20;
		}
		else
		{
			_capability = DeviceCapability::Sm10;
		}

		// Fetch some device statistics
		int nrMultiprocessors = 0;
		VCL_CU_SAFE_CALL(cuDeviceGetAttribute(&nrMultiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
		_nrComputeUnits = nrMultiprocessors;

		int nrAsyncEngines = 0;
		VCL_CU_SAFE_CALL(cuDeviceGetAttribute(&nrAsyncEngines, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev));
		_nrAsyncEngines = nrAsyncEngines;
	}

	bool Device::supports(Feature f) const
	{
		switch (f)
		{
		case Feature::DynamicParallelism:
			if (_capability == DeviceCapability::Sm35 || _capability == DeviceCapability::Sm50)
				return true;

			break;
		}

		return false;
	}
}}}
