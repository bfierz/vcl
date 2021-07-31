/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/descriptortable.h>
#include <vcl/graphics/d3d12/semaphore.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>

// Google test
#include <gtest/gtest.h>

extern std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

TEST(D3D12Buffer, Create)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	BufferDescription desc = {
		1024,
		BufferUsage::Vertex
	};

	D3D12::Buffer buf(device.get(), desc);

	// Verify the result
	EXPECT_TRUE(buf.handle() != 0) << "Buffer not created.";
}

/*TEST(D3D12Buffer, Clear)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	BufferDescription desc =
	{
		1024,
		ResourceUsage::Staging,
		{}
	};
	
	std::vector<int> read_back(1024 / sizeof(int), 0xDEADC0DE);
	BufferInitData data =
	{
		read_back.data(),
		1024
	};

	D3D12::Buffer buf(desc, false, false, &data);
	buf.clear();

	buf.copyTo(read_back.data(), 0, 0, 1024);
	
	// Verify the result
	bool equal = true;
	int fault = 0;
	for (int i : read_back) {
		equal = equal && (i == 0);
		if (i != 0)
			fault = i;
	}
	EXPECT_TRUE(equal) << "Buffer not cleared: " << std::hex << "0x" << fault;
}

TEST(D3D12Buffer, SetValue)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	BufferDescription desc =
	{
		1024,
		ResourceUsage::Default,
		{}
	};

	D3D12::Buffer buf(desc);

	int ref_data = 0xDEADC0DE;
	buf.clear(0, 1024, Vcl::Graphics::D3D12::RenderType<int>{}, &ref_data);

	std::vector<int> read_back(1024 / sizeof(int));
	buf.copyTo(read_back.data(), 0, 0, 1024);

	// Verify the result
	bool equal = true;
	int fault = 0;
	for (int i : read_back) {
		equal = equal && (i == ref_data);
		if (i != ref_data)
			fault = i;
	}
	EXPECT_TRUE(equal) << "Buffer not set. Ref: 0x" << std::hex << ref_data << ", actual: 0x" << fault;
}*/

TEST(D3D12Buffer, InitWithValues)
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

	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list_ptr = cmd_list.Get();
	D3D12::Buffer buf(device.get(), desc, &data, cmd_list_ptr);
	VCL_DIRECT3D_SAFE_CALL(cmd_list->Close());

	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	ID3D12CommandList* const generic_list = cmd_list_ptr;
	cmd_queue->ExecuteCommandLists(1, &generic_list);

	Vcl::Graphics::D3D12::Semaphore sema{ device->nativeDevice() };
	const auto sig_value = sema.signal(cmd_queue.Get());
	sema.wait(sig_value);

	bool equal = true;
	auto ptr = (float*)buf.map({ 0, 1024 });
	for (int i = 0; i < 256; i++)
	{
		equal = equal && (ptr[i] == numbers[i]);
	}
	buf.unmap({ 0, 0 });

	// Verify the result
	EXPECT_TRUE(buf.handle() != 0) << "Buffer not created.";
	EXPECT_TRUE(equal) << "Initialisation data is correct.";
}

TEST(D3D12Buffer, ReadWrite)
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

	D3D12::Buffer buf0(device.get(), desc);

	EXPECT_TRUE(buf0.handle() != 0) << "Buffer not created.";

	auto writePtr = (float*)buf0.map({ 0, 0 });
	for (int i = 0; i < 95; i++)
	{
		writePtr[i] = numbers[i];
	}
	buf0.unmap({ 0, 512 });

	auto readPtr = (float*)buf0.map({ 0, 1024 });
	bool equal = true;
	for (int i = 0; i < 95; i++)
	{
		equal = equal && (readPtr[i] == numbers[i]);
	}
	for (int i = 95; i < 256; i++)
	{
		equal = equal && (readPtr[i] == 0);
	}
	buf0.unmap({ 0, 0 });
	EXPECT_TRUE(equal) << "Initialisation data is correct.";
}

TEST(D3D12Buffer, DoubleMap)
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

	D3D12::Buffer buf0(device.get(), desc);

	EXPECT_TRUE(buf0.handle() != 0) << "Buffer not created.";

	auto writePtr0 = (float*)buf0.map({ 0, 0 });
	auto writePtr1 = (float*)buf0.map({ 0, 0 });
	for (int i = 0; i < 50; i++)
	{
		writePtr0[i] = numbers[i];
	}
	for (int i = 50; i < 95; i++)
	{
		writePtr1[i] = numbers[i];
	}
	buf0.unmap({ 0, 512 });
	buf0.unmap({ 512, 1024 });

	auto readPtr = (float*)buf0.map({ 0, 1024 });
	bool equal = true;
	for (int i = 0; i < 95; i++)
	{
		equal = equal && (readPtr[i] == numbers[i]);
	}
	for (int i = 95; i < 256; i++)
	{
		equal = equal && (readPtr[i] == 0);
	}
	buf0.unmap({ 0, 0 });
	EXPECT_TRUE(equal) << "Initialisation data is correct.";
}

TEST(D3D12Buffer, DynamicUpdate)
{
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	float numbers[256];
	for (int i = 0; i < 256; i++)
		numbers[i] = (float)i;

	float zeros[256];
	for (int i = 0; i < 256; i++)
		zeros[i] = 0;

	std::vector<DescriptorTableLayoutEntry> dynamic_resources =
	{
		{ DescriptorTableLayoutEntryType::Table, TableDescriptor{{
			{ D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
			}}, D3D12_SHADER_VISIBILITY_ALL }
	};

	DescriptorTableLayout table_layout{ device.get(), std::move(dynamic_resources), {} };
	auto signature = table_layout.rootSignature();

	BufferDescription desc = {
		1024,
		BufferUsage::MapWrite | BufferUsage::Uniform
	};

	D3D12::Buffer buf0(device.get(), desc);
	EXPECT_TRUE(buf0.handle() != 0) << "Buffer not created.";

	DescriptorTable table{ device.get(), &table_layout };
	table.addResource(0, &buf0, 0, 256, 4);

	auto writePtr = (float*)buf0.map();
	for (int i = 0; i < 95; i++)
	{
		writePtr[i] = numbers[i];
	}

	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);

	cmd_list->SetGraphicsRootSignature(signature);
	table.setToGraphics(cmd_list.Get(), 0);

	cmd_list->Close();
	ID3D12CommandList* const generic_list = cmd_list.Get();
	cmd_queue->ExecuteCommandLists(1, &generic_list);

	Vcl::Graphics::D3D12::Semaphore sema{ device->nativeDevice() };
	const auto sig_value = sema.signal(cmd_queue.Get());
	sema.wait(sig_value);
}
