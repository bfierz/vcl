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
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/descriptorset.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>

// Test
#include "test.h"

// Shaders
#include "quad.vert.spv.h"
const stdext::span<const uint32_t> QuadSpirvVS32{ reinterpret_cast<uint32_t*>(QuadSpirvVSData), QuadSpirvVSDataSize / 4 };

class VulkanDescriptorSetsTest : public VulkanTest
{
};

static Vcl::Graphics::Vulkan::DescriptorSet::BufferView makeView(const Vcl::Graphics::Runtime::Vulkan::Buffer& buffer)
{
	return {buffer.id(), 0, static_cast<uint32_t>(buffer.sizeInBytes())};
}

TEST(VulkanDescriptorSets, MergeBindingsDisjoint)
{
	using namespace Vcl::Graphics::Runtime::Vulkan;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	std::vector<DescriptorSetLayoutBinding> bindings0 =
	{
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 0 },
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 1 }
	};

	std::vector<DescriptorSetLayoutBinding> bindings1 =
	{
		{ DescriptorType::UniformBuffer, 1, ShaderStage::Vertex, 3 },
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 4 }
	};

	const auto merged = merge(bindings0, bindings1);
	EXPECT_EQ(merged.size(), 4);
	EXPECT_EQ(merged.at(0).Type, DescriptorType::StorageBuffer);
	EXPECT_EQ(merged.at(0).Binding, 0);
	EXPECT_EQ(merged.at(1).Type, DescriptorType::StorageBuffer);
	EXPECT_EQ(merged.at(1).Binding, 1);
	EXPECT_EQ(merged.at(2).Type, DescriptorType::UniformBuffer);
	EXPECT_EQ(merged.at(2).Binding, 3);
	EXPECT_EQ(merged.at(3).Type, DescriptorType::StorageBuffer);
	EXPECT_EQ(merged.at(3).Binding, 4);
}

TEST(VulkanDescriptorSets, MergeBindingsOverlapSame)
{
	using namespace Vcl::Graphics::Runtime::Vulkan;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	std::vector<DescriptorSetLayoutBinding> bindings0 =
	{
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 0 },
		{ DescriptorType::UniformBuffer, 1, ShaderStage::Compute, 1 }
	};

	std::vector<DescriptorSetLayoutBinding> bindings1 =
	{
		{ DescriptorType::UniformBuffer, 1, ShaderStage::Compute, 1 },
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 4 }
	};

	const auto merged = merge(bindings0, bindings1);
	EXPECT_EQ(merged.size(), 3);
	EXPECT_EQ(merged.at(0).Type, DescriptorType::StorageBuffer);
	EXPECT_EQ(merged.at(0).Binding, 0);
	EXPECT_EQ(merged.at(1).Type, DescriptorType::UniformBuffer);
	EXPECT_EQ(merged.at(1).Binding, 1);
	EXPECT_EQ(merged.at(2).Type, DescriptorType::StorageBuffer);
	EXPECT_EQ(merged.at(2).Binding, 4);
}

TEST(VulkanDescriptorSets, MergeBindingsIncompatible)
{
	using namespace Vcl::Graphics::Runtime::Vulkan;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	std::vector<DescriptorSetLayoutBinding> bindings0 =
	{
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 0 },
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 1 }
	};

	std::vector<DescriptorSetLayoutBinding> bindings1 =
	{
		{ DescriptorType::UniformBuffer, 1, ShaderStage::Vertex, 1 },
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 4 }
	};

	const auto merged = merge(bindings0, bindings1);
	EXPECT_EQ(merged.size(), 0);
}

TEST(VulkanDescriptorSets, MergeBindingsOverlapCompatible)
{
	using namespace Vcl::Graphics::Runtime::Vulkan;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	std::vector<DescriptorSetLayoutBinding> bindings0 =
	{
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 0 },
		{ DescriptorType::UniformBuffer, 1, ShaderStage::Compute, 1 }
	};

	std::vector<DescriptorSetLayoutBinding> bindings1 =
	{
		{ DescriptorType::UniformBuffer, 1, ShaderStage::Vertex, 1 },
		{ DescriptorType::StorageBuffer, 1, ShaderStage::Compute, 4 }
	};

	const auto merged = merge(bindings0, bindings1);
	EXPECT_EQ(merged.size(), 3);
	EXPECT_EQ(merged.at(0).Type, DescriptorType::StorageBuffer);
	EXPECT_EQ(merged.at(0).Binding, 0);
	EXPECT_EQ(merged.at(1).ShaderStages, Vcl::Flags<ShaderStage>({ ShaderStage::Vertex, ShaderStage::Compute }));
	EXPECT_EQ(merged.at(1).Type, DescriptorType::UniformBuffer);
	EXPECT_EQ(merged.at(1).Binding, 1);
	EXPECT_EQ(merged.at(2).Type, DescriptorType::StorageBuffer);
	EXPECT_EQ(merged.at(2).Binding, 4);
}

TEST_F(VulkanDescriptorSetsTest, Descriptor)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	// Compile the shader stages
	Runtime::Vulkan::Shader vs(*_context, ShaderType::VertexShader, 0, QuadSpirvVS32);
	EXPECT_NE(vs.getCreateInfo().module, nullptr);
	EXPECT_EQ(vs.sets().size(), 1);

	// Layout
	const auto& bindings = vs.bindings(0);
	DescriptorSetLayout descriptor_set_layout{ _context.get(), bindings };
	EXPECT_NE((VkDescriptorSetLayout)descriptor_set_layout, nullptr);

	// Descriptor set
	DescriptorSet desc_set{ &descriptor_set_layout };
	EXPECT_NE((VkDescriptorSet)desc_set, nullptr);

	// Fill descriptor set
	BufferDescription buffer_desc =
	{
		1024,
		ResourceUsage::Dynamic,
		{}
	};
	std::vector<float> buffer_init(256, 1.0f);
	BufferInitData buffer_init_dat =
	{
		buffer_init.data(),
		buffer_init.size() * sizeof(float)
	};
	Runtime::Vulkan::Buffer b0(_context.get(), buffer_desc, Runtime::Vulkan::BufferUsage::UniformBuffer, &buffer_init_dat);
	Runtime::Vulkan::Buffer b1(_context.get(), buffer_desc, Runtime::Vulkan::BufferUsage::UniformBuffer, &buffer_init_dat);
	Runtime::Vulkan::Buffer b2(_context.get(), buffer_desc, Runtime::Vulkan::BufferUsage::StorageBuffer, &buffer_init_dat);
	Runtime::Vulkan::Buffer b3(_context.get(), buffer_desc, Runtime::Vulkan::BufferUsage::StorageBuffer, &buffer_init_dat);

	std::vector<DescriptorSet::UpdateDescriptor> descriptors =
	{
		{ DescriptorType::UniformBuffer, 0, 0, makeView(b0) },
		{ DescriptorType::UniformBuffer, 1, 0, makeView(b1) },
		{ DescriptorType::StorageBuffer, 2, 0, makeView(b2) },
		{ DescriptorType::StorageBuffer, 2, 1, makeView(b3) },
	};
	desc_set.update(descriptors);
}
