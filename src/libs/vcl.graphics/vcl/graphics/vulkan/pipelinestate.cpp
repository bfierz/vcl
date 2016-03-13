/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#include <vcl/graphics/vulkan/pipelinestate.h>

// C++ standard library
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/vulkan/tools.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	DescriptorSetLayout::DescriptorSetLayout(Context* context, std::initializer_list<DescriptorSetLayoutBinding> bindings)
	{
		std::vector<VkDescriptorSetLayoutBinding> vk_bindings;
		vk_bindings.reserve(bindings.size());

		for (const auto& binding : bindings)
		{
			VkDescriptorSetLayoutBinding layout = {};
			layout.descriptorType = binding.Type;
			layout.stageFlags = binding.StageFlags;
			layout.binding = binding.Binding;
			layout.descriptorCount = 1;
			layout.pImmutableSamplers = nullptr;

			vk_bindings.push_back(layout);
		}

		VkDescriptorSetLayoutCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		info.pNext = nullptr;
		info.flags = 0;
		info.bindingCount = vk_bindings.size();
		info.pBindings = vk_bindings.data();

		VkResult res = vkCreateDescriptorSetLayout(*context, &info, NULL, &_layout);
		VclCheck(res == VK_SUCCESS, "Pipeline state was created.");
	}

	DescriptorSetLayout::~DescriptorSetLayout()
	{
		vkDestroyDescriptorSetLayout(*_context, _layout, nullptr);
	}

	VkPipelineVertexInputStateCreateInfo createVertexInputStateInfo
	(
		const Vcl::Graphics::Runtime::InputLayoutDescription& desc,
		std::vector<VkVertexInputBindingDescription>& bindings,
		std::vector<VkVertexInputAttributeDescription>& attributes
	)
	{
		VkPipelineVertexInputStateCreateInfo vi;
		vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vi.pNext = nullptr;
		vi.flags = 0;
		vi.vertexBindingDescriptionCount = desc.bindings().size();
		vi.vertexAttributeDescriptionCount = desc.attributes().size();

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
		for (const auto& attrib: desc.attributes())
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

	PipelineLayout::PipelineLayout(Context* context, DescriptorSetLayout* descriptor_set_layout)
	: _context(context)
	, _descriptorSetLayout(descriptor_set_layout)
	{
		VkPipelineLayoutCreateInfo info;
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		info.pNext = nullptr;
		info.flags = 0;
		info.setLayoutCount = 1;
		info.pSetLayouts = _descriptorSetLayout->ptr();
		info.pushConstantRangeCount = 0;
		info.pPushConstantRanges = nullptr;

		VkResult res = vkCreatePipelineLayout(*_context, &info, nullptr, &_layout);
		VclCheck(res == VK_SUCCESS, "Pipeline layout was created.");
	}

	PipelineLayout::~PipelineLayout()
	{
		vkDestroyPipelineLayout(*_context, _layout, nullptr);
	}

	PipelineState::PipelineState
	(
		Context* context, PipelineLayout* layout, VkRenderPass pass,
		const Vcl::Graphics::Runtime::PipelineStateDescription& desc
	)
	: _context(context)
	, _layout(layout)
	{
		VkGraphicsPipelineCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		info.layout = *_layout;
		info.renderPass = pass;

		// Convert the input layout
		const auto& input_layout = desc.InputLayout;
		std::vector<VkVertexInputBindingDescription> bindings;
		std::vector<VkVertexInputAttributeDescription> attributes;
		VkPipelineVertexInputStateCreateInfo vi = createVertexInputStateInfo(input_layout, bindings, attributes);
		info.pVertexInputState = &vi;

		// Configure the input assembly stage
		VkPipelineInputAssemblyStateCreateInfo ia;
		ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		ia.pNext = nullptr;
		ia.flags = 0;
		ia.topology = convert(desc.InputAssembly.Topology);
		ia.primitiveRestartEnable = desc.InputAssembly.PrimitiveRestartEnable;
		info.pInputAssemblyState = &ia;

		// Configure the rasterization state
		VkPipelineRasterizationStateCreateInfo rs = {};
		rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		info.pRasterizationState = &rs;

		// Configure the viewports
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;
		info.pViewportState = &viewportState;

		// Configure shaders
		VkPipelineShaderStageCreateInfo sh_info[5];
		int sh_idx = 0;
		if (desc.VertexShader && dynamic_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.VertexShader))
		{
			auto sh = static_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.VertexShader);
			sh_info[sh_idx++] = sh->getCreateInfo();
		}
		if (desc.TessControlShader && dynamic_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.TessControlShader))
		{
			auto sh = static_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.TessControlShader);
			sh_info[sh_idx++] = sh->getCreateInfo();
		}
		if (desc.TessEvalShader && dynamic_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.TessEvalShader))
		{
			auto sh = static_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.TessEvalShader);
			sh_info[sh_idx++] = sh->getCreateInfo();
		}
		if (desc.GeometryShader && dynamic_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.GeometryShader))
		{
			auto sh = static_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.GeometryShader);
			sh_info[sh_idx++] = sh->getCreateInfo();
		}
		if (desc.FragmentShader && dynamic_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.FragmentShader))
		{
			auto sh = static_cast<Vcl::Graphics::Runtime::Vulkan::Shader*>(desc.FragmentShader);
			sh_info[sh_idx++] = sh->getCreateInfo();
		}
		info.stageCount = sh_idx;
		info.pStages = sh_info;

		// Conifgure dynamic states
		VkPipelineDynamicStateCreateInfo dynamicState = {};
		VkDynamicState dynamicStateEnables[] =
		{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
		};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.pDynamicStates = dynamicStateEnables;
		dynamicState.dynamicStateCount = sizeof(dynamicStateEnables) / sizeof(VkDynamicState);
		info.pDynamicState = &dynamicState;

		VkResult res = vkCreateGraphicsPipelines(*_context, _context->cache(), 1, &info, nullptr, &_state);
		VclCheck(res == VK_SUCCESS, "Pipeline state was created.");
	}

	PipelineState::~PipelineState()
	{
		vkDestroyPipeline(*_context, _state, nullptr);
	}
}}}
