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
//#include <vcl/graphics/runtime/d3d12/state/blendstate.h>
//#include <vcl/graphics/runtime/d3d12/state/depthstencilstate.h>
//#include <vcl/graphics/runtime/d3d12/state/inputlayout.h>
//#include <vcl/graphics/runtime/d3d12/state/rasterizerstate.h>
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

		WGPURenderPipelineDescriptor graphics_pipeline_desc = {};

		WGPUPipelineLayoutDescriptor layout_desc = {};
		graphics_pipeline_desc.layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);
		graphics_pipeline_desc.vertexStage = getProgammableStageDesc(desc.VertexShader);
		const auto frag_shader_stage = getProgammableStageDesc(desc.FragmentShader);
		graphics_pipeline_desc.fragmentStage = &frag_shader_stage;
		//WGPUVertexStateDescriptor const* vertexState;
		graphics_pipeline_desc.primitiveTopology = convert(desc.InputAssembly.Topology);
		//WGPURasterizationStateDescriptor const* rasterizationState;
		graphics_pipeline_desc.sampleCount = 1;
		//WGPUDepthStencilStateDescriptor const* depthStencilState;

		std::array<WGPUColorStateDescriptor, 8> colour_states = {};
		colour_states[0].format = toWebGPUEnum(rt_formats.ColourFormats[0]);
		colour_states[0].alphaBlend.operation = WGPUBlendOperation_Add;
		colour_states[0].alphaBlend.srcFactor = WGPUBlendFactor_One;
		colour_states[0].alphaBlend.dstFactor = WGPUBlendFactor_Zero;
		colour_states[0].colorBlend.operation = WGPUBlendOperation_Add;
		colour_states[0].colorBlend.srcFactor = WGPUBlendFactor_One;
		colour_states[0].colorBlend.dstFactor = WGPUBlendFactor_Zero;
		colour_states[0].writeMask = WGPUColorWriteMask_All;

		graphics_pipeline_desc.colorStateCount = 1;
		graphics_pipeline_desc.colorStates = colour_states.data();
		graphics_pipeline_desc.sampleMask = UINT_MAX;
		graphics_pipeline_desc.alphaToCoverageEnabled = desc.Blend.AlphaToCoverageEnable;




		//graphics_pipeline_desc.BlendState = toD3D12(desc.Blend);
		//graphics_pipeline_desc.RasterizerState = toD3D12(desc.Rasterizer);
		//graphics_pipeline_desc.DepthStencilState = toD3D12(desc.DepthStencil);

		//const auto input_layout = toD3D12(desc.InputLayout);
		//graphics_pipeline_desc.InputLayout.NumElements = input_layout.size();
		//graphics_pipeline_desc.InputLayout.pInputElementDescs = input_layout.data();
		
		//graphics_pipeline_desc.IBStripCutValue = desc.InputAssembly.PrimitiveRestartEnable ? D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF : D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
		//if (rt_layout)
		//{
		//	graphics_pipeline_desc.NumRenderTargets = rt_layout->NumRenderTargets;
		//	memcpy(graphics_pipeline_desc.RTVFormats, rt_layout->RTVFormats, rt_layout->NumRenderTargets*sizeof(DXGI_FORMAT));
		//	graphics_pipeline_desc.DSVFormat = rt_layout->DSVFormat;
		//}
		//graphics_pipeline_desc.SampleDesc = { 1, 0 };

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
