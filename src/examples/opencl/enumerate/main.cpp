/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <iostream>

// VCL
#include <vcl/compute/opencl/context.h>
#include <vcl/compute/opencl/device.h>
#include <vcl/compute/opencl/platform.h>

int main(int argc, char* argv[])
{
	using namespace Vcl::Compute::OpenCL;

	Platform::initialise();

	unsigned int ui = 0;
	for (const auto& plat : Platform::instance()->availablePlatforms())
	{
		std::cout << ui << ": OpenCL name: "      << plat.Name << std::endl;
		std::cout << ui << ": OpenCL profile: "   << plat.Profile << std::endl;
		std::cout << ui << ": OpenCL version: "   << plat.Version << std::endl;
		std::cout << ui << ": OpenCL vendor: "    << plat.Vendor << std::endl;

		std::cout << ui << ": OpenCL extensions:" << std::endl;
		for (const auto& ext : plat.Extensions)
			std::cout << "\t" << ext << std::endl;

		std::cout << std::endl;

		// Increment platform index
		ui++;
	}

	std::cout << "Devices:" << std::endl;
	for (int d = 0; d < Platform::instance()->nrDevices(); d++)
	{
		auto& dev = Platform::instance()->device(d);

		std::cout << d << ": Name: " << dev.name() << std::endl;
		std::cout << d << ": Number of Compute Units: " << dev.nrComputeUnits() << std::endl;

		std::cout << std::endl;
	}

	Platform::dispose();

	return 0;
}
