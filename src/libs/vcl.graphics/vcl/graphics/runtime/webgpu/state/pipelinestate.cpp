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
#include <vcl/graphics/runtime/webgpu/state/pipelinestate.h>

// C++ Standard Library

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/webgpu/resource/shader.h>
#include <vcl/graphics/runtime/webgpu/state/blendstate.h>
#include <vcl/graphics/runtime/webgpu/state/depthstencilstate.h>
#include <vcl/graphics/runtime/webgpu/state/inputlayout.h>
#include <vcl/graphics/runtime/webgpu/state/rasterizerstate.h>
#include <vcl/graphics/webgpu/webgpu.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU
{
	WGPUPrimitiveTopology convert(PrimitiveType type)
	{
		switch (type)
		{
		case PrimitiveType::Undefined:        return WGPUPrimitiveTopology_Force32;
		case PrimitiveType::Pointlist:        return WGPUPrimitiveTopology_PointList;
		case PrimitiveType::Linelist:         return WGPUPrimitiveTopology_LineList;
		case PrimitiveType::Linestrip:        return WGPUPrimitiveTopology_LineStrip;
		case PrimitiveType::LinelistAdj:      return WGPUPrimitiveTopology_Force32;
		case PrimitiveType::LinestripAdj:     return WGPUPrimitiveTopology_Force32;
		case PrimitiveType::Trianglelist:     return WGPUPrimitiveTopology_TriangleList;
		case PrimitiveType::Trianglestrip:    return WGPUPrimitiveTopology_TriangleStrip;
		case PrimitiveType::TrianglelistAdj:  return WGPUPrimitiveTopology_Force32;
		case PrimitiveType::TrianglestripAdj: return WGPUPrimitiveTopology_Force32;
		case PrimitiveType::Patch:            return WGPUPrimitiveTopology_Force32;
		}

		return WGPUPrimitiveTopology_Force32;
	}

	bool isListFormat(PrimitiveType type)
	{
		switch (type)
		{
		case PrimitiveType::Pointlist:
		case PrimitiveType::Linelist:
		case PrimitiveType::Trianglelist:
			return true;
		default:
			return false;
		}
	}

	WGPUProgrammableStageDescriptor getProgammableStageDesc(const Runtime::Shader* shader)
	{
		VclRequire(dynamic_cast<const Shader*>(shader), "Shader is WebGPU shader");

		WGPUProgrammableStageDescriptor desc = {};

		const auto* wgpu_shader = static_cast<const Shader*>(shader);
		desc.module =  wgpu_shader->handle();
		desc.entryPoint = "main";

		return desc;
	}

	GraphicsPipelineState::GraphicsPipelineState
	(
		WGPUDevice device,
		const PipelineStateDescription& desc,
		const RenderTargetLayout& rt_formats
	)
	: _inputLayout{desc.InputLayout}
	{
		using Vcl::Graphics::WebGPU::toWebGPUEnum;

		VclRequire(desc.InputAssembly.PrimitiveRestartEnable == false, "Primitive restart is not supported.");
		VclRequire(desc.TessControlShader == nullptr, "Tessellation Control Shader not supported");
		VclRequire(desc.TessEvalShader == nullptr, "Tessellation Evaluation Shader not supported");
		VclRequire(desc.GeometryShader == nullptr, "Geometry Shader not supported");
		VclRequire(desc.Blend.IndependentBlendEnable, "WebGPU requires independent blending")
		VclRequire(desc.Rasterizer.MultisampleEnable == false, "Setting not supported")

		WGPURenderPipelineDescriptor graphics_pipeline_desc = {};

		WGPUPipelineLayoutDescriptor layout_desc = {};
		graphics_pipeline_desc.layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);
		graphics_pipeline_desc.vertexStage = getProgammableStageDesc(desc.VertexShader);
		const auto frag_shader_stage = getProgammableStageDesc(desc.FragmentShader);
		graphics_pipeline_desc.fragmentStage = &frag_shader_stage;

		WGPUVertexStateDescriptor vertex_state_desc = {};
		vertex_state_desc.indexFormat = isListFormat(desc.InputAssembly.Topology) ? WGPUIndexFormat_Undefined : WGPUIndexFormat_Uint32;
		auto vertex_buffer_desc = toWebGPU(desc.InputLayout);
		auto* attrib_base = vertex_buffer_desc.second.data();
		for (auto& vb : vertex_buffer_desc.first)
		{
			vb.attributes = attrib_base;
			attrib_base += vb.attributeCount;
		}
		vertex_state_desc.vertexBufferCount = vertex_buffer_desc.first.size();
		vertex_state_desc.vertexBuffers = vertex_buffer_desc.first.data();
		graphics_pipeline_desc.vertexState = &vertex_state_desc;

		graphics_pipeline_desc.primitiveTopology = convert(desc.InputAssembly.Topology);
		const auto rasterization_state = toWebGPU(desc.Rasterizer);
		graphics_pipeline_desc.rasterizationState = &rasterization_state;
		graphics_pipeline_desc.sampleCount = 1;
		auto depth_stencil_state = toWebGPU(desc.DepthStencil);
		depth_stencil_state.format = toWebGPUEnum(rt_formats.DepthStencilFormat);
		if (rt_formats.DepthStencilFormat == SurfaceFormat::Unknown)
			graphics_pipeline_desc.depthStencilState = nullptr;
		else
			graphics_pipeline_desc.depthStencilState = &depth_stencil_state;

		auto colour_states = toWebGPU(desc.Blend);
		for (size_t i = 0; i < rt_formats.ColourFormats.size(); i++)
		{
			colour_states[i].format = toWebGPUEnum(rt_formats.ColourFormats[i]);
		}
		graphics_pipeline_desc.colorStateCount = rt_formats.ColourFormats.size();
		graphics_pipeline_desc.colorStates = colour_states.data();
		graphics_pipeline_desc.sampleMask = UINT_MAX;
		graphics_pipeline_desc.alphaToCoverageEnabled = desc.Blend.AlphaToCoverageEnable;

		_pipeline = wgpuDeviceCreateRenderPipeline(device, &graphics_pipeline_desc);
	}

	ComputePipelineState::ComputePipelineState
	(
		WGPUDevice device,
		const ComputePipelineStateDescription& desc
	)
	{
		VclRequire(desc.ComputeShader, "Shader is set.");
		VclRequire(implies(desc.ComputeShader, desc.ComputeShader->type() == ShaderType::ComputeShader), "Shader is compute shader");

		WGPUComputePipelineDescriptor compute_pipeline_desc = {};
		compute_pipeline_desc.computeStage = getProgammableStageDesc(desc.ComputeShader);

		_pipeline = wgpuDeviceCreateComputePipeline(device, &compute_pipeline_desc);
	}
}}}}
