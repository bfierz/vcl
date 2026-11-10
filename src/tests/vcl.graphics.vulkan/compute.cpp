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
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/runtime/vulkan/state/pipelinestate.h>
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/descriptorheap.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>

// Test
#include "test.h"

// Additional shaders
#include "saxpy.comp.spv.h"
const stdext::span<const uint32_t> SaxpySpirvCS32{ reinterpret_cast<uint32_t*>(SaxpySpirvCSData), SaxpySpirvCSSize / 4 };

class VulkanComputeTest : public VulkanTest
{
};

struct KernelParameters
{
	// Total width
	uint32_t Width;

	// Total height
	uint32_t Height;

	// Saxpy scale 'a'
	float a;

	// Index of the array 'x' in the resource heap
	uint32_t X;

	// Index of the array 'y' in the resource heap
	uint32_t Y;
};

TEST_F(VulkanComputeTest, ComputeShader)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	// Compile the shader stages
	Runtime::Vulkan::Shader cs(*_context.get(), ShaderType::ComputeShader, 0, SaxpySpirvCS32);
	EXPECT_TRUE(cs.getCreateInfo().module != 0);
	EXPECT_EQ(cs.pushConstantSize(), sizeof(KernelParameters));

	// Bindless heap, which defines the layout of all pipelines
	DescriptorHeap heap{ _context.get(), {} };

	// Configure the pipeline
	ComputePipelineStateDescription pps_desc;
	pps_desc.ComputeShader = &cs;
	Runtime::Vulkan::ComputePipelineState pps{ _context.get(), &heap, pps_desc };

	// Buffer for the test
	BufferDescription b0_desc = {
		1024,
		Runtime::BufferUsage::MapWrite | Runtime::BufferUsage::CopySrc | Runtime::BufferUsage::Storage
	};
	BufferDescription b1_desc = {
		1024,
		Runtime::BufferUsage::MapRead | Runtime::BufferUsage::CopyDst | Runtime::BufferUsage::Storage
	};
	std::vector<float> buffer_init(256, 1.0f);
	BufferInitData buffer_init_dat = {
		buffer_init.data(),
		buffer_init.size() * sizeof(float)
	};
	Runtime::Vulkan::Buffer b0(_context.get(), b0_desc, &buffer_init_dat);
	Runtime::Vulkan::Buffer b1(_context.get(), b1_desc);

	// Occupy the first slot, such that the shader has to use the indices
	const ResourceView unused = heap.createView(b0, ResourceViewType::Buffer, 0, 16);
	const ResourceView x = heap.createView(b0, ResourceViewType::Buffer, 0, b0.sizeInBytes());
	const ResourceView y = heap.createView(b1, ResourceViewType::RWBuffer, 0, b1.sizeInBytes());
	EXPECT_NE(x.index().Value, y.index().Value);

	// Actual execution
	CommandQueue queue(_context.get(), VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT, 0);
	CommandBuffer cmd_buffer{ *_context, _context->commandPool(0, CommandBufferType::Transient) };
	cmd_buffer.begin();
	VkBufferCopy cpy_rgn = { 0, 0, 1024 };
	cmd_buffer.copy(b0.id(), b1.id(), { &cpy_rgn, 1 });
	heap.bind(cmd_buffer);
	cmd_buffer.bind(VK_PIPELINE_BIND_POINT_COMPUTE, pps);
	cmd_buffer.pushConstants<KernelParameters>(pps.layout(), 16u, 16u, 1.0f, x.index().Value, y.index().Value);
	cmd_buffer.dispatch(16, 16);
	cmd_buffer.end();
	queue.submit(cmd_buffer, VK_NULL_HANDLE);
	queue.waitIdle();

	std::vector<float> read_back(1024 / sizeof(float), 0);
	void* mapped = b1.memory()->map(0, 1024);
	memcpy(read_back.data(), mapped, 1024);
	b1.memory()->unmap();

	// Verify the result
	bool equal = true;
	int fault = 0;
	for (float i : read_back)
	{
		equal = equal && (i == 2.0f);
	}
	EXPECT_TRUE(equal);
}
