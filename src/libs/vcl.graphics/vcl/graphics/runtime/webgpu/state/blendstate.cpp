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
#include <vcl/graphics/runtime/webgpu/state/blendstate.h>

// C++ standard library
#include <utility>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU {
	WGPUBlendOperation toWebGPU(BlendOperation op)
	{
		// clang-format off
		switch (op) {
		case BlendOperation::Add        :  return WGPUBlendOperation_Add;
		case BlendOperation::Subtract   :  return WGPUBlendOperation_Subtract;
		case BlendOperation::RevSubtract:  return WGPUBlendOperation_ReverseSubtract;
		case BlendOperation::Min        :  return WGPUBlendOperation_Min;
		case BlendOperation::Max        :  return WGPUBlendOperation_Max;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return WGPUBlendOperation_Force32;
	}

	WGPUBlendFactor toWebGPU(Blend factor)
	{
		// clang-format off
		switch (factor) {
		case Blend::Zero          : return WGPUBlendFactor_Zero;
		case Blend::One           : return WGPUBlendFactor_One;
		case Blend::SrcColour     : return WGPUBlendFactor_Src;
		case Blend::InvSrcColour  : return WGPUBlendFactor_OneMinusSrc;
		case Blend::SrcAlpha      : return WGPUBlendFactor_SrcAlpha;
		case Blend::InvSrcAlpha   : return WGPUBlendFactor_OneMinusSrcAlpha;
		case Blend::DestAlpha     : return WGPUBlendFactor_DstAlpha;
		case Blend::InvDestAlpha  : return WGPUBlendFactor_OneMinusDstAlpha;
		case Blend::DestColour    : return WGPUBlendFactor_Dst;
		case Blend::InvDestColour : return WGPUBlendFactor_OneMinusDst;
		case Blend::SrcAlphaSat   : return WGPUBlendFactor_SrcAlphaSaturated;
		case Blend::BlendFactor   : return WGPUBlendFactor_Constant;
		case Blend::InvBlendFactor: return WGPUBlendFactor_OneMinusConstant;
		case Blend::Src1Colour:
		case Blend::InvSrc1Colour:
		case Blend::Src1Alpha:
		case Blend::InvSrc1Alpha:
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return WGPUBlendFactor_Force32;
	}

	std::array<WGPUBlendState, 8> toWebGPU(const BlendDescription& desc)
	{
		VclRequire(desc.LogicOpEnable == false, "WebGPU does not support logic ops");

		VclRequire(desc.RenderTarget[0].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported");
		VclRequire(desc.RenderTarget[1].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported");
		VclRequire(desc.RenderTarget[2].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported");
		VclRequire(desc.RenderTarget[3].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported");
		VclRequire(desc.RenderTarget[4].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported");
		VclRequire(desc.RenderTarget[5].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported");
		VclRequire(desc.RenderTarget[6].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported");
		VclRequire(desc.RenderTarget[7].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported");

		std::array<WGPUBlendState, 8> webgpu_desc = {};
		int i = 0;
		for (const auto& rt : desc.RenderTarget)
		{
			auto& tgt = webgpu_desc[i];
			if (rt.BlendEnable)
			{
				tgt.alpha.operation = toWebGPU(rt.BlendOpAlpha);
				tgt.alpha.srcFactor = toWebGPU(rt.SrcBlendAlpha);
				tgt.alpha.dstFactor = toWebGPU(rt.DestBlendAlpha);
				tgt.color.operation = toWebGPU(rt.BlendOp);
				tgt.color.srcFactor = toWebGPU(rt.SrcBlend);
				tgt.color.dstFactor = toWebGPU(rt.DestBlend);
			} else
			{
				tgt.alpha.operation = WGPUBlendOperation_Add;
				tgt.alpha.srcFactor = WGPUBlendFactor_One;
				tgt.alpha.dstFactor = WGPUBlendFactor_Zero;
				tgt.color.operation = WGPUBlendOperation_Add;
				tgt.color.srcFactor = WGPUBlendFactor_One;
				tgt.color.dstFactor = WGPUBlendFactor_Zero;
			}
			i++;
		}

		return webgpu_desc;
	}
}}}}
