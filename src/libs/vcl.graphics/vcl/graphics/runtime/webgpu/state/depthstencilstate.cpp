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
#include <vcl/graphics/runtime/webgpu/state/depthstencilstate.h>

// C++ standard library
#include <utility>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU
{
	WGPUCompareFunction toWebGPU(ComparisonFunction op)
	{
		switch (op)
		{
		case ComparisonFunction::Never:        return WGPUCompareFunction_Never;
		case ComparisonFunction::Less:         return WGPUCompareFunction_Less;
		case ComparisonFunction::Equal:        return WGPUCompareFunction_Equal;
		case ComparisonFunction::LessEqual:    return WGPUCompareFunction_LessEqual;
		case ComparisonFunction::Greater:      return WGPUCompareFunction_Greater;
		case ComparisonFunction::NotEqual:     return WGPUCompareFunction_NotEqual;
		case ComparisonFunction::GreaterEqual: return WGPUCompareFunction_GreaterEqual;
		case ComparisonFunction::Always:       return WGPUCompareFunction_Always;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return WGPUCompareFunction_Undefined;
	}

	WGPUStencilOperation toWebGPU(StencilOperation op)
	{
		switch (op)
		{
		case StencilOperation::Keep:             return WGPUStencilOperation_Keep;
		case StencilOperation::Zero:             return WGPUStencilOperation_Zero;
		case StencilOperation::Replace:          return WGPUStencilOperation_Replace;
		case StencilOperation::IncreaseSaturate: return WGPUStencilOperation_IncrementClamp;
		case StencilOperation::DecreaseSaturate: return WGPUStencilOperation_DecrementClamp;
		case StencilOperation::Invert:           return WGPUStencilOperation_Invert;
		case StencilOperation::IncreaseWrap:     return WGPUStencilOperation_IncrementWrap;
		case StencilOperation::DecreaseWrap:     return WGPUStencilOperation_DecrementWrap;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return WGPUStencilOperation_Force32;
	}

	WGPUStencilStateFaceDescriptor toWebGPU(DepthStencilOperationDescription desc)
	{
		WGPUStencilStateFaceDescriptor webgpu_desc = {};
		webgpu_desc.failOp = toWebGPU(desc.StencilFailOp);
		webgpu_desc.depthFailOp = toWebGPU(desc.StencilDepthFailOp);
		webgpu_desc.passOp = toWebGPU(desc.StencilPassOp);
		webgpu_desc.compare = toWebGPU(desc.StencilFunc);

		return webgpu_desc;
	}

	WGPUDepthStencilState toWebGPU(const DepthStencilDescription& desc)
	{
		VclRequire(desc.DepthEnable, "WebGPU requires enabled depth test");

		WGPUDepthStencilState webgpu_desc = {};

		webgpu_desc.depthWriteEnabled = desc.DepthWriteMask == DepthWriteMethod::All ? true : false;
		webgpu_desc.depthCompare = toWebGPU(desc.DepthFunc);
		webgpu_desc.stencilReadMask = desc.StencilReadMask;
		webgpu_desc.stencilWriteMask = desc.StencilWriteMask;
		if (desc.StencilEnable)
		{
			webgpu_desc.stencilFront = toWebGPU(desc.FrontFace);
			webgpu_desc.stencilBack = toWebGPU(desc.BackFace);
		}
		else
		{
			webgpu_desc.stencilBack.compare      = WGPUCompareFunction_Always;
			webgpu_desc.stencilBack.failOp       = WGPUStencilOperation_Keep;
			webgpu_desc.stencilBack.depthFailOp  = WGPUStencilOperation_Keep;
			webgpu_desc.stencilBack.passOp       = WGPUStencilOperation_Keep;
			webgpu_desc.stencilFront.compare     = WGPUCompareFunction_Always;
			webgpu_desc.stencilFront.failOp      = WGPUStencilOperation_Keep;
			webgpu_desc.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
			webgpu_desc.stencilFront.passOp      = WGPUStencilOperation_Keep;
		}

		return webgpu_desc;
	}
}}}}
