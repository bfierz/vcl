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
#include <vcl/config/cuda.h>

// C++ standard library
#include <string>

namespace Vcl { namespace Compute { namespace Cuda
{
	enum class DeviceCapability
	{
		Sm10,
		Sm20,
		Sm30,
		Sm35,
		Sm50
	};

	enum class Feature
	{
		DynamicParallelism
	};

	class Device
	{
	public:
		//! Constructor
		Device(CUdevice dev);

		//! Destructor
		virtual ~Device() = default;

		//! Convert to OpenCL device ID
		inline operator CUdevice() const
		{
			return _device;
		}

		//! \returns the name of this device
		const std::string& name() const { return _name; }

		//! \returns the capability level
		DeviceCapability capability() const { return _capability; }

		//! \returns the number of compute units
		uint32_t nrComputeUnits() const { return _nrComputeUnits; }

		//! \returns true when the queried feature is supported
		bool supports(Feature feature) const;

	private:
		//! CUDA device ID
		CUdevice _device{ 0 };

		//! Name of this device
		std::string _name;

		//! Device capability
		DeviceCapability _capability{ DeviceCapability::Sm10 };

		//! Number of compute units
		uint32_t _nrComputeUnits{ 0 };

		//! Number of asynchronous engines
		uint32_t _nrAsyncEngines{ 0 };
	};
}}}
