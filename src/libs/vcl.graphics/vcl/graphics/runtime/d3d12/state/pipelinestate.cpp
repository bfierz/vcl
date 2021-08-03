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
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>

// C++ Standard Library

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>
#include <vcl/graphics/runtime/d3d12/state/blendstate.h>
#include <vcl/graphics/runtime/d3d12/state/depthstencilstate.h>
#include <vcl/graphics/runtime/d3d12/state/inputlayout.h>
#include <vcl/graphics/runtime/d3d12/state/rasterizerstate.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 {
	D3D12_PRIMITIVE_TOPOLOGY_TYPE convert(PrimitiveType type)
	{
		switch (type)
		{
		case PrimitiveType::Undefined: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
		case PrimitiveType::Pointlist: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
		case PrimitiveType::Linelist:
		case PrimitiveType::Linestrip:
		case PrimitiveType::LinelistAdj:
		case PrimitiveType::LinestripAdj: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
		case PrimitiveType::Trianglelist:
		case PrimitiveType::Trianglestrip:
		case PrimitiveType::TrianglelistAdj:
		case PrimitiveType::TrianglestripAdj: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		case PrimitiveType::Patch: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
		}

		return D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
	}

	D3D12_SHADER_BYTECODE getShaderCode(const Runtime::Shader* shader)
	{
		if (!shader)
			return {};

		auto d3d12_shader = static_cast<const D3D12::Shader*>(shader);
		const auto cso = d3d12_shader->data();
		return { cso.data(), cso.size() };
	}

	GraphicsPipelineState::GraphicsPipelineState(
		Graphics::D3D12::Device* device,
		const PipelineStateDescription& desc,
		const RenderTargetLayout& rt_layout,
		const Graphics::D3D12::DescriptorTableLayout* layout)
	: _inputLayout{ desc.InputLayout }
	{
		VclRequire(desc.InputAssembly.PrimitiveRestartEnable == false, "Primitive restart is not supported.");

		D3D12_GRAPHICS_PIPELINE_STATE_DESC graphics_pipeline_desc = {};
		if (layout)
			graphics_pipeline_desc.pRootSignature = layout->rootSignature();

		graphics_pipeline_desc.VS = getShaderCode(desc.VertexShader);
		graphics_pipeline_desc.PS = getShaderCode(desc.FragmentShader);
		graphics_pipeline_desc.DS = getShaderCode(desc.TessControlShader);
		graphics_pipeline_desc.HS = getShaderCode(desc.TessEvalShader);
		graphics_pipeline_desc.GS = getShaderCode(desc.GeometryShader);
		//D3D12_STREAM_OUTPUT_DESC StreamOutput;

		graphics_pipeline_desc.BlendState = toD3D12(desc.Blend);
		graphics_pipeline_desc.SampleMask = UINT_MAX;
		graphics_pipeline_desc.RasterizerState = toD3D12(desc.Rasterizer);
		graphics_pipeline_desc.DepthStencilState = toD3D12(desc.DepthStencil);

		const auto input_layout = toD3D12(desc.InputLayout);
		graphics_pipeline_desc.InputLayout.NumElements = input_layout.size();
		graphics_pipeline_desc.InputLayout.pInputElementDescs = input_layout.data();

		graphics_pipeline_desc.IBStripCutValue = desc.InputAssembly.PrimitiveRestartEnable ? D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF : D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
		graphics_pipeline_desc.PrimitiveTopologyType = convert(desc.InputAssembly.Topology);
		graphics_pipeline_desc.NumRenderTargets = rt_layout.ColourFormats.size();
		for (size_t i = 0; i < rt_layout.ColourFormats.size(); i++)
		{
			graphics_pipeline_desc.RTVFormats[i] = Graphics::D3D12::D3D::toD3Denum(rt_layout.ColourFormats[i]);
		}
		graphics_pipeline_desc.DSVFormat = Graphics::D3D12::D3D::toD3Denum(rt_layout.DepthStencilFormat);
		graphics_pipeline_desc.SampleDesc = { 1, 0 };
		//UINT NodeMask;
		//D3D12_CACHED_PIPELINE_STATE CachedPSO;
		//D3D12_PIPELINE_STATE_FLAGS Flags;

		_pipeline = device->createGraphicsPipelineState(graphics_pipeline_desc);
	}

	ComputePipelineState::ComputePipelineState(
		Graphics::D3D12::Device* device,
		const ComputePipelineStateDescription& desc,
		const Graphics::D3D12::DescriptorTableLayout* layout)
	{
		VclRequire(desc.ComputeShader, "Shader is set.");
		VclRequire(implies(desc.ComputeShader, desc.ComputeShader->type() == ShaderType::ComputeShader), "Shader is compute shader");

		D3D12_COMPUTE_PIPELINE_STATE_DESC compute_pipeline_desc = {};
		compute_pipeline_desc.pRootSignature = layout ? layout->rootSignature() : nullptr;
		compute_pipeline_desc.CS = getShaderCode(desc.ComputeShader);

		_pipeline = device->createComputePipelineState(compute_pipeline_desc);
	}
}}}}
