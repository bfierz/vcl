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
#include <vcl/graphics/runtime/d3d12/state/depthstencilstate.h>

// C++ standard library
#include <utility>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 {
	D3D12_COMPARISON_FUNC toD3D12(ComparisonFunction op)
	{
		switch (op)
		{
		case ComparisonFunction::Never:        return D3D12_COMPARISON_FUNC_NEVER;
		case ComparisonFunction::Less:         return D3D12_COMPARISON_FUNC_LESS;
		case ComparisonFunction::Equal:        return D3D12_COMPARISON_FUNC_EQUAL;
		case ComparisonFunction::LessEqual:    return D3D12_COMPARISON_FUNC_LESS_EQUAL;
		case ComparisonFunction::Greater:      return D3D12_COMPARISON_FUNC_GREATER;
		case ComparisonFunction::NotEqual:     return D3D12_COMPARISON_FUNC_NOT_EQUAL;
		case ComparisonFunction::GreaterEqual: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
		case ComparisonFunction::Always:       return D3D12_COMPARISON_FUNC_ALWAYS;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return {};
	}

	D3D12_STENCIL_OP toD3D12(StencilOperation op)
	{
		switch (op)
		{
		case StencilOperation::Keep:             return D3D12_STENCIL_OP_KEEP;
		case StencilOperation::Zero:             return D3D12_STENCIL_OP_ZERO;
		case StencilOperation::Replace:          return D3D12_STENCIL_OP_REPLACE;
		case StencilOperation::IncreaseSaturate: return D3D12_STENCIL_OP_INCR_SAT;
		case StencilOperation::DecreaseSaturate: return D3D12_STENCIL_OP_DECR_SAT;
		case StencilOperation::Invert:           return D3D12_STENCIL_OP_INVERT;
		case StencilOperation::IncreaseWrap:     return D3D12_STENCIL_OP_INCR;
		case StencilOperation::DecreaseWrap:     return D3D12_STENCIL_OP_DECR;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return {};
	}

	D3D12_DEPTH_STENCILOP_DESC toD3D12(DepthStencilOperationDescription desc)
	{
		D3D12_DEPTH_STENCILOP_DESC d3d12_desc;
		d3d12_desc.StencilFailOp = toD3D12(desc.StencilFailOp);
		d3d12_desc.StencilDepthFailOp = toD3D12(desc.StencilDepthFailOp);
		d3d12_desc.StencilPassOp = toD3D12(desc.StencilPassOp);
		d3d12_desc.StencilFunc = toD3D12(desc.StencilFunc);

		return d3d12_desc;
	}

	D3D12_DEPTH_STENCIL_DESC toD3D12(const DepthStencilDescription& desc)
	{
		D3D12_DEPTH_STENCIL_DESC d3d12_desc;
		d3d12_desc.DepthEnable = desc.DepthEnable;
		d3d12_desc.DepthWriteMask = desc.DepthWriteMask == DepthWriteMethod::All ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
		d3d12_desc.DepthFunc = toD3D12(desc.DepthFunc);
		d3d12_desc.StencilEnable = desc.StencilEnable;
		d3d12_desc.StencilReadMask = desc.StencilReadMask;
		d3d12_desc.StencilWriteMask = desc.StencilWriteMask;
		d3d12_desc.FrontFace = toD3D12(desc.FrontFace);
		d3d12_desc.BackFace = toD3D12(desc.BackFace);

		return d3d12_desc;
	}
}}}}
