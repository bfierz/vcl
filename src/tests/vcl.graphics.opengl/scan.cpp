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

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/graphics/opengl/algorithm/scan.h>

// Google test
#include <gtest/gtest.h>

void ExecuteScanTest(unsigned int size)
{
	using namespace Vcl::Graphics;

	ScanExclusive scan{ size };

	// Define the input buffer
	std::vector<int> numbers(size);
	for (int i = 0; i < size; i++)
		numbers[i] = i;

	Runtime::BufferDescription desc = {
		static_cast<unsigned int>(sizeof(unsigned int) * numbers.size()),
		Runtime::BufferUsage::MapRead | Runtime::BufferUsage::Storage
	};

	Runtime::BufferInitData data = {
		numbers.data(),
		static_cast<unsigned int>(sizeof(unsigned int) * numbers.size())
	};

	auto input = Vcl::make_owner<Runtime::OpenGL::Buffer>(desc, &data);
	auto output = Vcl::make_owner<Runtime::OpenGL::Buffer>(desc);

	scan(output, input, size);

	int* ptr = (int*)output->map(0, sizeof(unsigned int) * numbers.size());

	int s = 0;
	for (int i = 0; i < size; i++)
	{
		EXPECT_EQ(s, ptr[i]) << "Prefix sum is wrong: " << i;
		s += i;
	}

	output->unmap();
}

TEST(OpenGL, ScanExclusiveSmall)
{
	ExecuteScanTest(4);
	ExecuteScanTest(12);
	ExecuteScanTest(16);
	ExecuteScanTest(40);
	ExecuteScanTest(256);
	ExecuteScanTest(1020);
	ExecuteScanTest(1024);
}

TEST(OpenGL, ScanExclusiveLarge)
{
	ExecuteScanTest(2048);
	ExecuteScanTest(3072);
}
