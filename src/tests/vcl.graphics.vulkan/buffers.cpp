/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <vcl/graphics/runtime/vulkan/resource/buffer.h>

// Include the relevant parts from the library
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>

// Test
#include "test.h"

class VulkanBufferTest : public VulkanTest
{
};

TEST_F(VulkanBufferTest, CreateBuffer)
{
	using namespace Vcl::Graphics::Runtime;

	// Define the buffer
	BufferDescription desc =
	{
		1024,
		BufferUsage::CopyDst
	};

	Vulkan::Buffer buf(_context.get(), desc);

	// Verify the result
	EXPECT_TRUE(buf.id()) << "Buffer not created.";
}

TEST_F(VulkanBufferTest, ClearBuffer)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;

	// Instiate command queue and buffer
	CommandQueue cmd_queue{ _context.get(), VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT, 0 };
	CommandBuffer cmd_buffer{ *_context, _context->commandPool(0, CommandBufferType::Transient) };

	// Define the source buffer
	BufferDescription desc =
	{
		1024,
		BufferUsage::MapRead | BufferUsage::CopyDst
	};
	Vulkan::Buffer buffer(_context.get(), desc);
	
	// Clear the buffer
	cmd_buffer.begin();
	cmd_buffer.fillBuffer(buffer.id(), 0, VK_WHOLE_SIZE, 0xDEADC0DE);
	cmd_buffer.end();

	std::array<VkCommandBuffer, 1> cmd_buffers = { cmd_buffer };
	cmd_queue.submit(stdext::make_span(cmd_buffers), VK_NULL_HANDLE);
	cmd_queue.waitIdle();

	std::vector<int> read_back(1024 / sizeof(int), 0);
	void* mapped = buffer.memory()->map(0, 1024);
	memcpy(read_back.data(), mapped, 1024);
	buffer.memory()->unmap();
	
	// Verify the result
	bool equal = true;
	int fault = 0;
	for (int i : read_back)
	{
		equal = equal && (i == 0xDEADC0DE);
		if (i != 0xDEADC0DE)
		{
			fault = i;
		}
	}
	EXPECT_TRUE(equal) << "Buffer not cleared: " << std::hex << "0x" << fault;
}

TEST_F(VulkanBufferTest, CopyBuffer)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;

	// Instiate command queue and buffer
	CommandQueue cmd_queue{ _context.get(), VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT, 0 };
	CommandBuffer cmd_buffer{ *_context, _context->commandPool(0, CommandBufferType::Transient) };

	// Define the source buffer
	BufferDescription write_buffer_desc =
	{
		1024,
		BufferUsage::MapWrite | BufferUsage::CopySrc | BufferUsage::CopyDst
	};
	Vulkan::Buffer write_buffer(_context.get(), write_buffer_desc);

	// Define the destination buffer
	BufferDescription read_buffer_desc =
	{
		1024,
		BufferUsage::MapRead | BufferUsage::CopyDst
	};
	Vulkan::Buffer read_buffer(_context.get(), read_buffer_desc);

	unsigned* mem = (unsigned*)write_buffer.memory()->map(0, 1024);
	for (int i = 0; i < 256; i++)
		mem[i] = 0xDEADC0DE;
	write_buffer.memory()->unmap();

	// Clear the buffer
	cmd_buffer.begin();

	VkBufferCopy cpy_rgn = {0, 0, 1024};
	cmd_buffer.copy(write_buffer.id(), read_buffer.id(), { &cpy_rgn, 1 });
	cmd_buffer.end();

	std::array<VkCommandBuffer, 1> cmd_buffers = { cmd_buffer };
	cmd_queue.submit(stdext::make_span(cmd_buffers), VK_NULL_HANDLE);
	cmd_queue.waitIdle();

	std::vector<int> read_back(1024 / sizeof(int), 0);
	void* mapped = read_buffer.memory()->map(0, 1024);
	memcpy(read_back.data(), mapped, 1024);
	read_buffer.memory()->unmap();

	// Verify the result
	bool equal = true;
	int fault = 0;
	for (int i : read_back)
	{
		equal = equal && (i == 0xDEADC0DE);
		if (i != 0xDEADC0DE)
		{
			fault = i;
		}
	}
	EXPECT_TRUE(equal) << "Buffer not cleared: " << std::hex << "0x" << fault;
}
