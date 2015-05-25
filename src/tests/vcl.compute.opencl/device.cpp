/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/compute/opencl/device.h>
#include <vcl/compute/opencl/platform.h>

// Google test
#include <gtest/gtest.h>

// Tests the scalar gather function.
TEST(OpenCL, QueryDevices)
{
	using namespace Vcl::Compute::OpenCL;

	Platform::initialise();

	//std::cout << ui << ": OpenCL name: " << buffer.data() << std::endl;
	//std::cout << ui << ": OpenCL profile: " << buffer.data() << std::endl;
	//std::cout << ui << ": OpenCL version: " << buffer.data() << std::endl;
	//std::cout << ui << ": OpenCL vendor: " << buffer.data() << std::endl;
	//std::cout << ui << ": OpenCL extensions:" << std::endl;
	//std::cout << "\t" << str_buffer.substr(head, tail - head) << std::endl;

	for (int d = 0; d < Platform::instance()->nrDevices(); d++)
	{
		auto& dev = Platform::instance()->device(d);
	}
}
