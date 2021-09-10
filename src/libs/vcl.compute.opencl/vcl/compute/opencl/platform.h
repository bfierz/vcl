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
#include <memory>
#include <string>
#include <vector>

// VCL
#include <vcl/compute/opencl/device.h>

namespace Vcl { namespace Compute { namespace OpenCL {
	struct PlatformDesc
	{
		cl_platform_id Id;
		std::string Name;
		std::string Profile;
		std::string Version;
		std::string Vendor;
		std::vector<std::string> Extensions;
	};

	class Platform
	{
	public:
		static void initialise();
		static Platform* instance();
		static void dispose();

	private:
		Platform();
		~Platform();

	public:
		const std::vector<PlatformDesc>& availablePlatforms() const;

	public:
		int nrDevices() const;
		const Device& device(int idx) const;

	private:
		std::vector<Device> _devices;
		std::vector<PlatformDesc> _platforms;

	private:
		static Platform* _implementation;
	};
}}}
