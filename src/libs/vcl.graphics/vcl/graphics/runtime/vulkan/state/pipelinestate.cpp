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
	namespace {
		//! Create a pipeline rasterizing primitives, which are generated either
		//! from vertex input or by a mesh shader (without vertex input state)
		VkPipeline createRasterPipeline(
			Vcl::Graphics::Vulkan::Context* context,
			VkPipelineLayout layout,
			const std::vector<VkPipelineShaderStageCreateInfo>& stages,
			const VkPipelineVertexInputStateCreateInfo* vertex_input,
			const VkPipelineInputAssemblyStateCreateInfo* input_assembly,
			const BlendDescription& blend_desc,
			const RasterizerDescription& rasterizer_desc,
			const DepthStencilDescription& depth_stencil_desc,
			const RenderTargetLayout& rt_layout)
		{
			using Vcl::Graphics::Vulkan::convert;
			using Vcl::Graphics::Vulkan::isStencilFormat;

			VclRequire(rt_layout.ColourFormats.size() <= blend_desc.RenderTarget.size(), "Number of render-targets is supported.");

			// Viewports and scissor rectangles are set dynamically
			VkPipelineViewportStateCreateInfo viewport = {};
			viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewport.viewportCount = 1;
			viewport.scissorCount = 1;

			const VkPipelineRasterizationStateCreateInfo rasterizer = toVulkan(rasterizer_desc);

			VkPipelineMultisampleStateCreateInfo multisample = {};
			multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
			multisample.alphaToCoverageEnable = blend_desc.AlphaToCoverageEnable ? VK_TRUE : VK_FALSE;

			const VkPipelineDepthStencilStateCreateInfo depth_stencil = toVulkan(depth_stencil_desc);

			std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(rt_layout.ColourFormats.size());
			const VkPipelineColorBlendStateCreateInfo blend = toVulkan(blend_desc, stdext::make_span(blend_attachments));

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
			vk_desc.pVertexInputState = vertex_input;
			vk_desc.pInputAssemblyState = input_assembly;
			vk_desc.pViewportState = &viewport;
			vk_desc.pRasterizationState = &rasterizer;
			vk_desc.pMultisampleState = &multisample;
			vk_desc.pDepthStencilState = &depth_stencil;
			vk_desc.pColorBlendState = &blend;
			vk_desc.pDynamicState = &dynamic_state;
			vk_desc.layout = layout;
			vk_desc.renderPass = VK_NULL_HANDLE;
			vk_desc.basePipelineHandle = VK_NULL_HANDLE;
			vk_desc.basePipelineIndex = -1;

			VkPipeline state = VK_NULL_HANDLE;
			const auto res = vkCreateGraphicsPipelines(*context, context->cache(), 1, &vk_desc, nullptr, &state);
			VclEnsure(res == VK_SUCCESS, "Pipeline state created");
			return state;
		}

		//! Collect the stage of a shader, if it is set
		void addStage(std::vector<VkPipelineShaderStageCreateInfo>& stages, const Runtime::Shader* shader, ShaderType type, const Vcl::Graphics::Vulkan::DescriptorHeap* heap)
		{
			if (!shader)
				return;

			VclRequire(dynamic_cast<const Runtime::Vulkan::Shader*>(shader), "Shader is Vulkan shader");
			VclRequire(shader->type() == type, "Shader has the type of its stage.");
			VclRequire(static_cast<const Runtime::Vulkan::Shader*>(shader)->pushConstantSize() <= heap->desc().PushConstantSize, "Push constants of the shader fit the layout.");
			VCL_UNREFERENCED_PARAMETER(type);
			VCL_UNREFERENCED_PARAMETER(heap);

			stages.push_back(static_cast<const Runtime::Vulkan::Shader*>(shader)->getCreateInfo());
		}
	}

	GraphicsPipelineState::GraphicsPipelineState(
		Vcl::Graphics::Vulkan::Context* context,
		const Vcl::Graphics::Vulkan::DescriptorHeap* heap,
		const PipelineStateDescription& desc)
	: _context(context)
	, _layout(heap ? heap->layout() : VK_NULL_HANDLE)
	, _inputLayout(desc.InputLayout)
	{
		using Vcl::Graphics::Vulkan::convert;

		VclRequire(heap, "Descriptor heap is set");
		VclRequire(validate(desc, context->features()).empty(), "Pipeline is supported by the device.");

		// Shader stages
		std::vector<VkPipelineShaderStageCreateInfo> stages;
		addStage(stages, desc.VertexShader, ShaderType::VertexShader, heap);
		addStage(stages, desc.GeometryShader, ShaderType::GeometryShader, heap);
		addStage(stages, desc.FragmentShader, ShaderType::FragmentShader, heap);

		// Vertex input
		std::vector<VkVertexInputBindingDescription> bindings;
		std::vector<VkVertexInputAttributeDescription> attributes;
		const VkPipelineVertexInputStateCreateInfo vertex_input = toVulkan(desc.InputLayout, bindings, attributes);

		VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = convert(desc.InputAssembly.Topology);
		input_assembly.primitiveRestartEnable = desc.InputAssembly.PrimitiveRestartEnable ? VK_TRUE : VK_FALSE;

		_state = createRasterPipeline(context, _layout, stages, &vertex_input, &input_assembly, desc.Blend, desc.Rasterizer, desc.DepthStencil, desc.RenderTargets);
	}

	GraphicsPipelineState::~GraphicsPipelineState()
	{
		if (_state)
			vkDestroyPipeline(*_context, _state, nullptr);
	}

	MeshPipelineState::MeshPipelineState(
		Vcl::Graphics::Vulkan::Context* context,
		const Vcl::Graphics::Vulkan::DescriptorHeap* heap,
		const MeshPipelineStateDescription& desc)
	: _context(context)
	, _layout(heap ? heap->layout() : VK_NULL_HANDLE)
	{
		VclRequire(heap, "Descriptor heap is set");
		VclRequire(validate(desc, context->features()).empty(), "Pipeline is supported by the device.");

		std::vector<VkPipelineShaderStageCreateInfo> stages;
		addStage(stages, desc.TaskShader, ShaderType::TaskShader, heap);
		addStage(stages, desc.MeshShader, ShaderType::MeshShader, heap);
		addStage(stages, desc.FragmentShader, ShaderType::FragmentShader, heap);

		_state = createRasterPipeline(context, _layout, stages, nullptr, nullptr, desc.Blend, desc.Rasterizer, desc.DepthStencil, desc.RenderTargets);
	}

	MeshPipelineState::~MeshPipelineState()
	{
		if (_state)
			vkDestroyPipeline(*_context, _state, nullptr);
	}

	ComputePipelineState::ComputePipelineState(
		Vcl::Graphics::Vulkan::Context* context,
		const Vcl::Graphics::Vulkan::DescriptorHeap* heap,
		const ComputePipelineStateDescription& desc)
	: _context(context)
	, _layout(heap ? heap->layout() : VK_NULL_HANDLE)
	{
		VclRequire(heap, "Descriptor heap is set");
		VclRequire(dynamic_cast<Runtime::Vulkan::Shader*>(desc.ComputeShader), "Shader is Vulkan shader");
		VclRequire(desc.ComputeShader->type() == ShaderType::ComputeShader, "Shader is compute shader");
		VclRequire(static_cast<Runtime::Vulkan::Shader*>(desc.ComputeShader)->pushConstantSize() <= heap->desc().PushConstantSize, "Push constants of the shader fit the layout.");

		// Create the corresponding Vulkan object
		VkComputePipelineCreateInfo vk_desc;
		vk_desc.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		vk_desc.pNext = nullptr;
		vk_desc.flags = 0;
		vk_desc.stage = static_cast<Runtime::Vulkan::Shader*>(desc.ComputeShader)->getCreateInfo();
		vk_desc.layout = _layout;
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
