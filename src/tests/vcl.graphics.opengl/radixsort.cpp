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
#include <vcl/graphics/opengl/algorithm/radixsort.h>

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

	Runtime::BufferDescription desc =
	{
		sizeof(unsigned int) * numbers.size(),
		Runtime::Usage::Staging,
		Runtime::CPUAccess::Read | Runtime::CPUAccess::Write
	};

	Runtime::BufferInitData data =
	{
		numbers.data(),
		sizeof(unsigned int) * numbers.size()
	};

	auto input = Vcl::make_owner<Runtime::OpenGL::Buffer>(desc, true, true, &data);
	auto output = Vcl::make_owner<Runtime::OpenGL::Buffer>(desc, true, true);

	scan(output, input, size);

	int* ptr = (int*)output->map(0, sizeof(unsigned int) * numbers.size(), Runtime::CPUAccess::Read);

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

void ExecuteRadixSortTest(unsigned int size)
{
	using namespace Vcl::Graphics;

	const unsigned int num_keys = size;

	RadixSort sort{ num_keys };

	// Define the input buffer
	std::vector<int> numbers(num_keys);
	for (int i = 0; i < num_keys; i++)
		numbers[i] = num_keys - (i + 0);

	Runtime::BufferDescription desc =
	{
		num_keys * sizeof(int),
		Runtime::Usage::Staging,
		Runtime::CPUAccess::Read | Runtime::CPUAccess::Write
	};

	Runtime::BufferInitData data =
	{
		numbers.data(),
		num_keys * sizeof(int)
	};

	auto keys = Vcl::make_owner<Runtime::OpenGL::Buffer>(desc, true, true, &data);

	sort(keys, num_keys, 20);

	int* ptr = (int*)keys->map(0, num_keys * sizeof(int), Runtime::CPUAccess::Read);

	int last = std::numeric_limits<int>::min();
	for (int i = 0; i < num_keys; i++)
	{
		EXPECT_LE(last, ptr[i]) << "Order is wrong: " << i;
		last = ptr[i];
	}

	keys->unmap();
}

TEST(OpenGL, RadixSort)
{
	ExecuteRadixSortTest(512);
	ExecuteRadixSortTest(768);
	ExecuteRadixSortTest(1024);
	ExecuteRadixSortTest(2048);
	ExecuteRadixSortTest(3072);
}
