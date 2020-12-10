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
#include <vcl/graphics/runtime/d3d12/state/rasterizerstate.h>

// C++ standard library
#include <utility>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12
{
	D3D12_CULL_MODE toD3D12(CullModeMethod op)
	{
		switch (op)
		{
		case CullModeMethod::None:  return D3D12_CULL_MODE_NONE;
		case CullModeMethod::Front: return D3D12_CULL_MODE_FRONT;
		case CullModeMethod::Back:  return D3D12_CULL_MODE_BACK;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return {};
	}

	D3D12_FILL_MODE toD3D12(FillModeMethod op)
	{
		switch (op)
		{
		case FillModeMethod::Solid:     return D3D12_FILL_MODE_SOLID;
		case FillModeMethod::Wireframe: return D3D12_FILL_MODE_WIREFRAME;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return {};
	}

	D3D12_RASTERIZER_DESC toD3D12(const RasterizerDescription& desc)
	{
		D3D12_RASTERIZER_DESC d3d12_desc;
		d3d12_desc.FillMode = toD3D12(desc.FillMode);
		d3d12_desc.CullMode = toD3D12(desc.CullMode);
		d3d12_desc.FrontCounterClockwise = desc.FrontCounterClockwise;
		d3d12_desc.DepthBias = desc.DepthBias;
		d3d12_desc.DepthBiasClamp = desc.SlopeScaledDepthBias;
		d3d12_desc.SlopeScaledDepthBias = desc.SlopeScaledDepthBias;
		d3d12_desc.DepthClipEnable = desc.DepthClipEnable;
		d3d12_desc.MultisampleEnable = desc.MultisampleEnable;
		d3d12_desc.AntialiasedLineEnable = desc.AntialiasedLineEnable;
		d3d12_desc.ForcedSampleCount = 0;
		d3d12_desc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
		//desc.ScissorEnable;

		return d3d12_desc;
	}
}}}}
