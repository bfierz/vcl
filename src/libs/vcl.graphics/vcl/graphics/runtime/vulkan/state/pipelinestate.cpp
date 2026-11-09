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

// C++ standard library
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/runtime/vulkan/state/blendstate.h>
#include <vcl/graphics/runtime/vulkan/state/depthstencilstate.h>
#include <vcl/graphics/runtime/vulkan/state/inputlayout.h>
#include <vcl/graphics/runtime/vulkan/state/rasterizerstate.h>
#include <vcl/graphics/vulkan/tools.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {
	PipelineLayout::PipelineLayout(
		Vcl::Graphics::Vulkan::Context* context,
		const DescriptorSetLayout* descriptor_set_layout,
		stdext::span<const PushConstantDescriptor> push_constants)
	: _context(context)
	, _descriptorSetLayout(descriptor_set_layout)
	{
		VkDescriptorSetLayout layouts[] = {
			*_descriptorSetLayout
		};

		std::vector<VkPushConstantRange> vk_push_constants;
		vk_push_constants.reserve(push_constants.size());
		std::transform(std::begin(push_constants), std::end(push_constants), std::back_inserter(vk_push_constants), [](const auto& c) {
			return VkPushConstantRange{ convert(c.StageFlags), c.Offset, c.Size };
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

	GraphicsPipelineState::GraphicsPipelineState(
		Vcl::Graphics::Vulkan::Context* context,
		const PipelineLayout* layout,
		const PipelineStateDescription& desc,
		const RenderTargetLayout& rt_layout)
	: _context(context)
	, _layout(layout)
	, _inputLayout(desc.InputLayout)
	{
		using Vcl::Graphics::Vulkan::convert;
		using Vcl::Graphics::Vulkan::isStencilFormat;

		VclRequire(layout, "Layout is set");
		VclRequire(desc.VertexShader, "Vertex shader is set");
		VclRequire(!desc.TessControlShader && !desc.TessEvalShader, "Tessellation requires the patch size, which is not part of the pipeline description.");
		VclRequire(rt_layout.ColourFormats.size() <= desc.Blend.RenderTarget.size(), "Number of render-targets is supported.");

		// Shader stages
		std::vector<VkPipelineShaderStageCreateInfo> stages;
		for (const auto* shader : { desc.VertexShader, desc.GeometryShader, desc.FragmentShader })
		{
			if (!shader)
				continue;

			VclRequire(dynamic_cast<const Runtime::Vulkan::Shader*>(shader), "Shader is Vulkan shader");
			stages.push_back(static_cast<const Runtime::Vulkan::Shader*>(shader)->getCreateInfo());
		}

		// Fixed function state
		std::vector<VkVertexInputBindingDescription> bindings;
		std::vector<VkVertexInputAttributeDescription> attributes;
		const VkPipelineVertexInputStateCreateInfo vertex_input = toVulkan(desc.InputLayout, bindings, attributes);

		VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = convert(desc.InputAssembly.Topology);
		input_assembly.primitiveRestartEnable = desc.InputAssembly.PrimitiveRestartEnable ? VK_TRUE : VK_FALSE;

		// Viewports and scissor rectangles are set dynamically
		VkPipelineViewportStateCreateInfo viewport = {};
		viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport.viewportCount = 1;
		viewport.scissorCount = 1;

		const VkPipelineRasterizationStateCreateInfo rasterizer = toVulkan(desc.Rasterizer);

		VkPipelineMultisampleStateCreateInfo multisample = {};
		multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample.alphaToCoverageEnable = desc.Blend.AlphaToCoverageEnable ? VK_TRUE : VK_FALSE;

		const VkPipelineDepthStencilStateCreateInfo depth_stencil = toVulkan(desc.DepthStencil);

		std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(rt_layout.ColourFormats.size());
		const VkPipelineColorBlendStateCreateInfo blend = toVulkan(desc.Blend, stdext::make_span(blend_attachments));

		const VkDynamicState dynamic_states[] = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
			VK_DYNAMIC_STATE_STENCIL_REFERENCE,
		};
		VkPipelineDynamicStateCreateInfo dynamic_state = {};
		dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_state.dynamicStateCount = static_cast<uint32_t>(sizeof(dynamic_states) / sizeof(VkDynamicState));
		dynamic_state.pDynamicStates = dynamic_states;

		// Render-target formats used with dynamic rendering
		std::vector<VkFormat> colour_formats;
		colour_formats.reserve(rt_layout.ColourFormats.size());
		for (const auto fmt : rt_layout.ColourFormats)
			colour_formats.push_back(convert(fmt));

		const VkFormat depth_format = rt_layout.DepthStencilFormat != SurfaceFormat::Unknown ? convert(rt_layout.DepthStencilFormat) : VK_FORMAT_UNDEFINED;

		VkPipelineRenderingCreateInfo rendering = {};
		rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
		rendering.colorAttachmentCount = static_cast<uint32_t>(colour_formats.size());
		rendering.pColorAttachmentFormats = colour_formats.data();
		rendering.depthAttachmentFormat = depth_format;
		rendering.stencilAttachmentFormat = isStencilFormat(depth_format) ? depth_format : VK_FORMAT_UNDEFINED;

		VkGraphicsPipelineCreateInfo vk_desc = {};
		vk_desc.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		vk_desc.pNext = &rendering;
		vk_desc.stageCount = static_cast<uint32_t>(stages.size());
		vk_desc.pStages = stages.data();
		vk_desc.pVertexInputState = &vertex_input;
		vk_desc.pInputAssemblyState = &input_assembly;
		vk_desc.pViewportState = &viewport;
		vk_desc.pRasterizationState = &rasterizer;
		vk_desc.pMultisampleState = &multisample;
		vk_desc.pDepthStencilState = &depth_stencil;
		vk_desc.pColorBlendState = &blend;
		vk_desc.pDynamicState = &dynamic_state;
		vk_desc.layout = *_layout;
		vk_desc.renderPass = VK_NULL_HANDLE;
		vk_desc.basePipelineHandle = VK_NULL_HANDLE;
		vk_desc.basePipelineIndex = -1;

		const auto res = vkCreateGraphicsPipelines(*context, context->cache(), 1, &vk_desc, nullptr, &_state);
		VclEnsure(res == VK_SUCCESS, "Pipeline state created");
	}

	GraphicsPipelineState::~GraphicsPipelineState()
	{
		if (_state)
			vkDestroyPipeline(*_context, _state, nullptr);
	}

	ComputePipelineState::ComputePipelineState(
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
