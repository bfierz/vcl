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
#include <vcl/graphics/runtime/d3d12/state/blendstate.h>

// C++ standard library
#include <utility>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/d3d.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 {

	D3D12_LOGIC_OP toD3D12(LogicOperation op)
	{
		switch (op)
		{
		case LogicOperation::Clear       : return D3D12_LOGIC_OP_CLEAR;
		case LogicOperation::Set         : return D3D12_LOGIC_OP_SET;
		case LogicOperation::Copy        : return D3D12_LOGIC_OP_COPY;
		case LogicOperation::CopyInverted: return D3D12_LOGIC_OP_COPY_INVERTED;
		case LogicOperation::NoOp        : return D3D12_LOGIC_OP_NOOP;
		case LogicOperation::Invert      : return D3D12_LOGIC_OP_INVERT;
		case LogicOperation::And         : return D3D12_LOGIC_OP_AND;
		case LogicOperation::Nand        : return D3D12_LOGIC_OP_NAND;
		case LogicOperation::Or          : return D3D12_LOGIC_OP_OR;
		case LogicOperation::Nor         : return D3D12_LOGIC_OP_NOR;
		case LogicOperation::Xor         : return D3D12_LOGIC_OP_XOR;
		case LogicOperation::Equiv       : return D3D12_LOGIC_OP_EQUIV;
		case LogicOperation::AndReverse  : return D3D12_LOGIC_OP_AND_REVERSE;
		case LogicOperation::AndInverted : return D3D12_LOGIC_OP_AND_INVERTED;
		case LogicOperation::OrReverse   : return D3D12_LOGIC_OP_OR_REVERSE;
		case LogicOperation::OrInverted  : return D3D12_LOGIC_OP_OR_INVERTED;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return {};
	}

	D3D12_BLEND_OP toD3D12(BlendOperation op)
	{
		switch (op)
		{
		case BlendOperation::Add        :  return D3D12_BLEND_OP_ADD;
		case BlendOperation::Subtract   :  return D3D12_BLEND_OP_SUBTRACT;
		case BlendOperation::RevSubtract:  return D3D12_BLEND_OP_REV_SUBTRACT;
		case BlendOperation::Min        :  return D3D12_BLEND_OP_MIN;
		case BlendOperation::Max        :  return D3D12_BLEND_OP_MAX;

		default: { VclDebugError("Enumeration value is valid."); }
		}

		return {};
	}

	D3D12_BLEND toD3D12(Blend factor)
	{
		switch (factor)
		{
		case Blend::Zero          : return D3D12_BLEND_ZERO;
		case Blend::One           : return D3D12_BLEND_ONE;
		case Blend::SrcColour     : return D3D12_BLEND_SRC_COLOR;
		case Blend::InvSrcColour  : return D3D12_BLEND_INV_SRC_COLOR;
		case Blend::SrcAlpha      : return D3D12_BLEND_SRC_ALPHA;
		case Blend::InvSrcAlpha   : return D3D12_BLEND_INV_SRC_ALPHA;
		case Blend::DestAlpha     : return D3D12_BLEND_DEST_ALPHA;
		case Blend::InvDestAlpha  : return D3D12_BLEND_INV_DEST_ALPHA;
		case Blend::DestColour    : return D3D12_BLEND_DEST_COLOR;
		case Blend::InvDestColour : return D3D12_BLEND_INV_DEST_COLOR;
		case Blend::SrcAlphaSat   : return D3D12_BLEND_SRC_ALPHA_SAT;
		case Blend::BlendFactor   : return D3D12_BLEND_BLEND_FACTOR;
		case Blend::InvBlendFactor: return D3D12_BLEND_INV_BLEND_FACTOR;
		case Blend::Src1Colour    : return D3D12_BLEND_SRC1_COLOR;
		case Blend::InvSrc1Colour : return D3D12_BLEND_INV_SRC1_COLOR;
		case Blend::Src1Alpha     : return D3D12_BLEND_SRC1_ALPHA;
		case Blend::InvSrc1Alpha  : return D3D12_BLEND_INV_SRC1_ALPHA;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return {};
	}

	D3D12_BLEND_DESC toD3D12(const BlendDescription& desc)
	{
		// Check consistency
		VclRequire(implies(desc.LogicOpEnable, desc.RenderTarget[0].BlendEnable == false && desc.IndependentBlendEnable == false), "Either logic ops or blend ops are enabled.");
		VclRequire(implies(desc.RenderTarget[0].BlendEnable, desc.LogicOpEnable == false), "Either logic ops or blend ops are enabled.");

		VclRequire(desc.RenderTarget[0].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported.");
		VclRequire(desc.RenderTarget[1].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported.");
		VclRequire(desc.RenderTarget[2].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported.");
		VclRequire(desc.RenderTarget[3].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported.");
		VclRequire(desc.RenderTarget[4].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported.");
		VclRequire(desc.RenderTarget[5].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported.");
		VclRequire(desc.RenderTarget[6].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported.");
		VclRequire(desc.RenderTarget[7].BlendOp < BlendOperation::Multiply, "Advanced blending operations are not supported.");

		D3D12_BLEND_DESC d3d12_desc;
		d3d12_desc.AlphaToCoverageEnable = desc.AlphaToCoverageEnable;
		d3d12_desc.IndependentBlendEnable = desc.IndependentBlendEnable;
		int i = 0;
		for (const auto& rt : desc.RenderTarget)
		{
			auto& tgt = d3d12_desc.RenderTarget[i];
			tgt.BlendEnable = rt.BlendEnable;
			tgt.LogicOpEnable = desc.LogicOpEnable;
			tgt.SrcBlend = toD3D12(rt.SrcBlend);
			tgt.DestBlend = toD3D12(rt.DestBlend);
			tgt.BlendOp = toD3D12(rt.BlendOp);
			tgt.SrcBlendAlpha = toD3D12(rt.SrcBlendAlpha);
			tgt.DestBlendAlpha = toD3D12(rt.DestBlendAlpha);
			tgt.BlendOpAlpha = toD3D12(rt.BlendOpAlpha);
			tgt.LogicOp = toD3D12(desc.LogicOp);
			tgt.RenderTargetWriteMask = rt.RenderTargetWriteMask.bits();
			i++;
		}

		return d3d12_desc;
	}
}}}}
