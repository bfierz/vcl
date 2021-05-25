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

		WGPUVertexState vertex_state_desc = {};
		auto vertex_buffer_desc = toWebGPU(desc.InputLayout);
		auto* attrib_base = vertex_buffer_desc.second.data();
		for (auto& vb : vertex_buffer_desc.first)
		{
			vb.attributes = attrib_base;
			attrib_base += vb.attributeCount;
		}
		vertex_state_desc.bufferCount = vertex_buffer_desc.first.size();
		vertex_state_desc.buffers = vertex_buffer_desc.first.data();
		const auto vertex_shader_desc = getProgammableStageDesc(desc.VertexShader);
		graphics_pipeline_desc.vertex.module = vertex_shader_desc.module;
		graphics_pipeline_desc.vertex.entryPoint = vertex_shader_desc.entryPoint;

		graphics_pipeline_desc.primitive.topology = convert(desc.InputAssembly.Topology);
		graphics_pipeline_desc.primitive.stripIndexFormat = isListFormat(desc.InputAssembly.Topology) ? WGPUIndexFormat_Undefined : WGPUIndexFormat_Uint32;
		graphics_pipeline_desc.primitive.frontFace = desc.Rasterizer.FrontCounterClockwise ? WGPUFrontFace_CCW : WGPUFrontFace_CW;;
		graphics_pipeline_desc.primitive.cullMode = toWebGPU(desc.Rasterizer.CullMode);
		graphics_pipeline_desc.multisample.count = 1;
		graphics_pipeline_desc.multisample.mask = UINT_MAX;
		graphics_pipeline_desc.multisample.alphaToCoverageEnabled = desc.Blend.AlphaToCoverageEnable;
		auto depth_stencil_state = toWebGPU(desc.DepthStencil);
		toWebGPU(desc.Rasterizer, &depth_stencil_state);
		depth_stencil_state.format = toWebGPUEnum(rt_formats.DepthStencilFormat);
		if (rt_formats.DepthStencilFormat == SurfaceFormat::Unknown)
			graphics_pipeline_desc.depthStencil = nullptr;
		else
			graphics_pipeline_desc.depthStencil = &depth_stencil_state;

		const auto frag_shader_stage = getProgammableStageDesc(desc.FragmentShader);
		std::array<WGPUColorTargetState, 8> colour_states = {};
		const auto blend_states = toWebGPU(desc.Blend);
		for (size_t i = 0; i < rt_formats.ColourFormats.size(); i++)
		{
			colour_states[i].format = toWebGPUEnum(rt_formats.ColourFormats[i]);
			colour_states[i].blend = &blend_states[i];
			colour_states[i].writeMask = desc.Blend.RenderTarget[i].RenderTargetWriteMask.bits();
		}

		WGPUFragmentState fragment_state = {};
		fragment_state.module = frag_shader_stage.module;
		fragment_state.entryPoint = frag_shader_stage.entryPoint;
		fragment_state.targetCount = rt_formats.ColourFormats.size();
		fragment_state.targets = colour_states.data();
		graphics_pipeline_desc.fragment = &fragment_state;

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
