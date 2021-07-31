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
#include <vcl/graphics/opengl/algorithm/radixsort.h>

// Google test
#include <gtest/gtest.h>

void ExecuteRadixSortTest(unsigned int size)
{
	using namespace Vcl::Graphics;

	const unsigned int num_keys = size;

	RadixSort sort{ num_keys };

	// Define the input buffer
	std::vector<int> numbers(num_keys);
	for (int i = 0; i < num_keys; i++)
		numbers[i] = num_keys - (i + 0);

	Runtime::BufferDescription desc = {
		num_keys * static_cast<unsigned int>(sizeof(int)),
		Runtime::BufferUsage::MapRead | Runtime::BufferUsage::Storage
	};

	Runtime::BufferInitData data = {
		numbers.data(),
		num_keys * static_cast<unsigned int>(sizeof(int))
	};

	auto keys = Vcl::make_owner<Runtime::OpenGL::Buffer>(desc, &data);

	sort(keys, num_keys, 20);

	int* ptr = (int*)keys->map(0, num_keys * sizeof(int));

	int last = std::numeric_limits<int>::min();
	for (int i = 0; i < num_keys; i++)
	{
		EXPECT_LT(last, ptr[i]) << "Order is wrong: " << i;
		last = ptr[i];
	}

	keys->unmap();
}

extern bool isLlvmPipe;

TEST(OpenGL, RadixSort)
{
	if (isLlvmPipe)
	{
		std::cout << "[ SKIPPED  ] Test does not work under LLVM-pipe" << std::endl;
		return;
	}

	// Test range of valid input sizes
	for (int i = 512; i < (1 << 14); i += 512)
	{
		ExecuteRadixSortTest(i);
	}
}
