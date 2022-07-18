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
#include <random>

// Include the relevant parts from the library
#include <vcl/graphics/opengl/algorithm/histogram.h>

// Google test
#include <gtest/gtest.h>

void BuildHistogramTest(unsigned int buckets, unsigned int size)
{
	using namespace Vcl::Graphics;

	// Random number generator
	std::mt19937 rnd;
	std::uniform_int_distribution<unsigned int> dist{ 0, buckets - 1 };

	Histogram hist{ size, buckets };

	// Define the input buffer
	std::vector<int> numbers(size);
	for (unsigned int i = 0; i < size; i++)
		numbers[i] = dist(rnd);

	std::vector<int> histogram(buckets, 0);
	for (unsigned int i = 0; i < size; i++)
		histogram[numbers[i]]++;

	Runtime::BufferDescription desc = {
		static_cast<unsigned int>(sizeof(unsigned int) * numbers.size()),
		Runtime::BufferUsage::Storage
	};

	Runtime::BufferInitData data = {
		numbers.data(),
		static_cast<unsigned int>(sizeof(unsigned int) * numbers.size())
	};

	Runtime::BufferDescription out_desc = {
		static_cast<unsigned int>(sizeof(unsigned int)) * buckets,
		Runtime::BufferUsage::MapRead | Runtime::BufferUsage::Storage
	};

	auto input = Vcl::make_owner<Runtime::OpenGL::Buffer>(desc, &data);
	auto output = Vcl::make_owner<Runtime::OpenGL::Buffer>(out_desc);

	hist(output, input, size);

	int* ptr = (int*)output->map(0, output->sizeInBytes());

	for (unsigned int i = 0; i < buckets; i++)
	{
		EXPECT_EQ(histogram[i], ptr[i]) << "Prefix sum is wrong: " << i;
	}

	output->unmap();
}

TEST(OpenGL, Histogram)
{
	BuildHistogramTest(64, 32);
	BuildHistogramTest(64, 64);
	BuildHistogramTest(64, 65);
	BuildHistogramTest(64, 128);
	BuildHistogramTest(64, 129);
	BuildHistogramTest(64, 512);
}
