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
#include <vcl/config/opencl.h>

// C++ standard library
#include <string>

namespace Vcl { namespace Compute { namespace OpenCL {
	class Device
	{
	public:
		//! Constructor
		Device(cl_device_id dev);

		//! Destructor
		virtual ~Device() = default;

		//! Convert to OpenCL device ID
		inline operator cl_device_id() const
		{
			return _device;
		}

		//! \returns the name of this device
		const std::string& name() const { return _name; }

		//! \returns the number of compute units
		uint32_t nrComputeUnits() const { return _nrComputeUnits; }

	private:
		//! OpenCL device ID
		cl_device_id _device;

		//! Name of this device
		std::string _name;

		//! Major version of the OpenCL API
		int _capMajor;

		//! Minor version of the OpenCL API
		int _capMinor;

		//! Number of compute units
		uint32_t _nrComputeUnits;
	};
}}}
