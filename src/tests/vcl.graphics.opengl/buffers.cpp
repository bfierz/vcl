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
#include <vcl/graphics/opengl/gl.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>

// Google test
#include <gtest/gtest.h>

TEST(OpenGL, CreateBuffer)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	BufferDescription desc = {
		1024,
		BufferUsage::Storage
	};

	OpenGL::Buffer buf(desc);

	// Verify the result
	EXPECT_TRUE(buf.id() != 0) << "Buffer not created.";
}

TEST(OpenGL, ClearBuffer)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	BufferDescription desc = {
		1024,
		BufferUsage::CopySrc | BufferUsage::MapRead
	};

	std::vector<int> read_back(1024 / sizeof(int), 0xDEADC0DE);
	BufferInitData data = {
		read_back.data(),
		1024
	};

	OpenGL::Buffer buf(desc, &data);
	buf.clear();
	buf.copyTo(read_back.data(), 0, 0, 1024);

	// Verify the result
	bool equal = true;
	int fault = 0;
	for (int i : read_back)
	{
		equal = equal && (i == 0);
		if (i != 0)
			fault = i;
	}
	EXPECT_TRUE(equal) << "Buffer not cleared: " << std::hex << "0x" << fault;
}

TEST(OpenGL, SetBufferValue)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	BufferDescription desc = {
		1024,
		BufferUsage::CopySrc | BufferUsage::MapRead
	};

	OpenGL::Buffer buf(desc);

	int ref_data = 0xDEADC0DE;
	buf.clear(0, 1024, Vcl::Graphics::OpenGL::RenderType<int>{}, &ref_data);

	std::vector<int> read_back(1024 / sizeof(int));
	buf.copyTo(read_back.data(), 0, 0, 1024);

	// Verify the result
	bool equal = true;
	int fault = 0;
	for (int i : read_back)
	{
		equal = equal && (i == ref_data);
		if (i != ref_data)
			fault = i;
	}
	EXPECT_TRUE(equal) << "Buffer not set. Ref: 0x" << std::hex << ref_data << ", actual: 0x" << fault;
}

TEST(OpenGL, CheckBufferInit)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	float numbers[256];
	for (int i = 0; i < 256; i++)
		numbers[i] = (float)i;

	BufferDescription desc = {
		1024,
		BufferUsage::MapRead
	};

	BufferInitData data = {
		numbers,
		1024
	};

	OpenGL::Buffer buf(desc, &data);

	bool equal = true;
	auto ptr = (float*)buf.map(0, 1024);
	for (int i = 0; i < 256; i++)
	{
		equal = equal && (ptr[i] == numbers[i]);
	}
	buf.unmap();

	// Verify the result
	EXPECT_TRUE(buf.id() != 0) << "Buffer not created.";
	EXPECT_TRUE(equal) << "Initialisation data is correct.";
}

TEST(OpenGL, CheckExplicitBufferReadWrite)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	float numbers[256];
	for (int i = 0; i < 256; i++)
		numbers[i] = (float)i;

	float zeros[256];
	for (int i = 0; i < 256; i++)
		zeros[i] = 0;

	BufferDescription desc = {
		1024,
		BufferUsage::MapWrite
	};

	BufferInitData data = {
		zeros,
		1024
	};

	OpenGL::Buffer buf0(desc, &data);

	EXPECT_TRUE(buf0.id() != 0) << "Buffer not created.";

	auto writePtr = (float*)buf0.map(0, 1024);
	for (int i = 0; i < 95; i++)
	{
		writePtr[i] = numbers[i];
	}
	buf0.unmap();

	auto readPtr = (float*)buf0.map(0, 1024);
	bool equal = true;
	for (int i = 0; i < 95; i++)
	{
		equal = equal && (readPtr[i] == numbers[i]);
	}
	for (int i = 95; i < 256; i++)
	{
		equal = equal && (readPtr[i] == 0);
	}
	buf0.unmap();
	EXPECT_TRUE(equal) << "Initialisation data is correct.";
}
