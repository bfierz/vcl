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
#include <vcl/graphics/vulkan/descriptorset.h>
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

static Vcl::Graphics::Vulkan::DescriptorSet::BufferView makeView(const Vcl::Graphics::Runtime::Vulkan::Buffer& buffer)
{
	return {buffer.id(), 0, static_cast<uint32_t>(buffer.sizeInBytes())};
}

struct KernelParameters
{
	// Total width
	uint32_t Width;

	// Total height
	uint32_t Height;

	// Saxpy scale 'a'
	float a;
};

TEST_F(VulkanComputeTest, ComputeShader)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	// Compile the shader stages
	Runtime::Vulkan::Shader cs(*_context.get(), ShaderType::ComputeShader, 0, SaxpySpirvCS32);
	EXPECT_TRUE(cs.getCreateInfo().module != 0);
	EXPECT_EQ(cs.sets().size(), 1);

	// Layout
	const auto& bindings = cs.bindings(0);
	const PushConstantDescriptor push_constant_descs[] =
	{
		cs.pushConstants()
	};
	DescriptorSetLayout descriptor_set_layout{_context.get(), bindings};
	Runtime::Vulkan::PipelineLayout layout{_context.get(), &descriptor_set_layout, push_constant_descs };

	// Configure the pipeline
	ComputePipelineStateDescription pps_desc;
	pps_desc.ComputeShader = &cs;
	Runtime::Vulkan::ComputePipelineState pps{_context.get(), &layout, pps_desc};

	// Descriptor sets
	DescriptorSet desc_set(&descriptor_set_layout);

	// Buffer for the test
	BufferDescription b0_desc =
	{
		1024,
		Runtime::BufferUsage::MapWrite | Runtime::BufferUsage::CopySrc | Runtime::BufferUsage::Storage
	};
	BufferDescription b1_desc = {
		1024,
		Runtime::BufferUsage::MapRead | Runtime::BufferUsage::CopyDst | Runtime::BufferUsage::Storage
	};
	std::vector<float> buffer_init(256, 1.0f);
	BufferInitData buffer_init_dat = 
	{
		buffer_init.data(),
		buffer_init.size() * sizeof(float)
	};
	Runtime::Vulkan::Buffer b0(_context.get(), b0_desc, &buffer_init_dat);
	Runtime::Vulkan::Buffer b1(_context.get(), b1_desc);

	std::vector<DescriptorSet::UpdateDescriptor> descriptors =
	{
		{ DescriptorType::StorageBuffer, 0, 0, makeView(b0) },
		{ DescriptorType::StorageBuffer, 1, 0, makeView(b1) }
	};
	desc_set.update(descriptors);

	// Actual execution
	CommandQueue queue(_context.get(), VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT, 0);
	CommandBuffer cmd_buffer{ *_context, _context->commandPool(0, CommandBufferType::Transient) };
	cmd_buffer.begin();
	VkBufferCopy cpy_rgn = { 0, 0, 1024 };
	cmd_buffer.copy(b0.id(), b1.id(), { &cpy_rgn, 1 });
	cmd_buffer.bind(desc_set, pps.layout());
	cmd_buffer.bind(pps);
	cmd_buffer.pushConstants<KernelParameters>(pps.layout(), ShaderStage::Compute, 16, 16, 1.0f);
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
