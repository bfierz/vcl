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
#include <vcl/graphics/runtime/vulkan/state/pipelinestate.h>

// VCL
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/vulkan/tools.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan
{
	VkPipelineVertexInputStateCreateInfo createVertexInputStateInfo
	(
		const Vcl::Graphics::Runtime::InputLayoutDescription& desc,
		std::vector<VkVertexInputBindingDescription>& bindings,
		std::vector<VkVertexInputAttributeDescription>& attributes
	)
	{
		using Vcl::Graphics::Vulkan::convert;

		VkPipelineVertexInputStateCreateInfo vi;
		vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vi.pNext = nullptr;
		vi.flags = 0;
		vi.vertexBindingDescriptionCount = static_cast<uint32_t>(desc.bindings().size());
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(desc.attributes().size());

		bindings.resize(desc.bindings().size());
		attributes.resize(desc.attributes().size());

		size_t idx = 0;
		for (const auto& binding : desc.bindings())
		{
			bindings[idx].binding = binding.Binding;
			bindings[idx].stride = binding.Stride;
			switch (binding.InputRate)
			{
			case Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerObject:
				bindings[idx].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
				break;
			case Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerInstance:
				bindings[idx].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
				break;
			}
			idx++;
		}
		vi.pVertexBindingDescriptions = bindings.data();

		idx = 0;
		for (const auto& attrib : desc.attributes())
		{
			attributes[idx].binding = attrib.InputSlot;
			attributes[idx].location = desc.location(idx);
			attributes[idx].format = convert(attrib.Format);
			attributes[idx].offset = attrib.Offset;
			idx++;
		}
		vi.pVertexAttributeDescriptions = attributes.data();

		return vi;
	}

	PipelineLayout::PipelineLayout
	(
		Vcl::Graphics::Vulkan::Context* context,
		const DescriptorSetLayout* descriptor_set_layout,
		stdext::span<const PushConstantDescriptor> push_constants
	)
	: _context(context)
	, _descriptorSetLayout(descriptor_set_layout)
	{
		VkDescriptorSetLayout layouts[] =
		{
			*_descriptorSetLayout
		};

		std::vector<VkPushConstantRange> vk_push_constants;
		vk_push_constants.reserve(push_constants.size());
		std::transform(std::begin(push_constants), std::end(push_constants),
			           std::back_inserter(vk_push_constants), [](const auto& c)
		{
			return VkPushConstantRange{convert(c.StageFlags), c.Offset, c.Size};
		});

		VkPipelineLayoutCreateInfo info;
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		info.pNext = nullptr;
		info.flags = 0;
		info.setLayoutCount = 1;
		info.pSetLayouts = layouts;
		info.pushConstantRangeCount =
			static_cast<uint32_t>(vk_push_constants.size());
		info.pPushConstantRanges = vk_push_constants.data();

		VkResult res = vkCreatePipelineLayout(*_context, &info, nullptr, &_layout);
		VclCheck(res == VK_SUCCESS, "Pipeline layout was created.");
	}

	PipelineLayout::PipelineLayout(PipelineLayout&& rhs)
	{
		std::swap(_context, rhs._context);
		std::swap(_descriptorSetLayout, rhs._descriptorSetLayout);
		std::swap(_layout, rhs._layout);
	}

	PipelineLayout::~PipelineLayout()
	{
		if (_layout)
		{
			vkDestroyPipelineLayout(*_context, _layout, nullptr);
		}
	}

	ComputePipelineState::ComputePipelineState
	(
		Vcl::Graphics::Vulkan::Context* context,
		const PipelineLayout* layout,
		const ComputePipelineStateDescription& desc)
	: _context(context)
	, _layout(layout)
	{
		VclRequire(layout, "Layout is set");
		VclRequire(dynamic_cast<Runtime::Vulkan::Shader*>(desc.ComputeShader), "Shader is Vulkan shader");
		VclRequire(desc.ComputeShader->type() == ShaderType::ComputeShader, "Shader is compute shader");

		// Create the corresponding Vulkan object
		VkComputePipelineCreateInfo vk_desc;
		vk_desc.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		vk_desc.pNext = nullptr;
		vk_desc.flags = 0;
		vk_desc.stage = static_cast<Runtime::Vulkan::Shader*>(desc.ComputeShader)->getCreateInfo();
		vk_desc.layout = *_layout;
		vk_desc.basePipelineHandle = VK_NULL_HANDLE;
		vk_desc.basePipelineIndex = -1;

		const auto res = vkCreateComputePipelines(*context, VK_NULL_HANDLE, 1, &vk_desc, nullptr, &_state);
		VclEnsure(res == VK_SUCCESS, "Pipeline state created");
	}

	ComputePipelineState::~ComputePipelineState()
	{
		if (_state)
			vkDestroyPipeline(*_context, _state, nullptr);
	}
}}}}
