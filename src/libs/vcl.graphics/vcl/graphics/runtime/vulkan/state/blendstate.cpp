/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2026 Basil Fierz
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
#include <vcl/graphics/runtime/vulkan/state/blendstate.h>

// C++ standard library
#include <algorithm>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {

	VkLogicOp toVulkan(LogicOperation op)
	{
		// clang-format off
		switch (op) {
		case LogicOperation::Clear       : return VK_LOGIC_OP_CLEAR;
		case LogicOperation::Set         : return VK_LOGIC_OP_SET;
		case LogicOperation::Copy        : return VK_LOGIC_OP_COPY;
		case LogicOperation::CopyInverted: return VK_LOGIC_OP_COPY_INVERTED;
		case LogicOperation::NoOp        : return VK_LOGIC_OP_NO_OP;
		case LogicOperation::Invert      : return VK_LOGIC_OP_INVERT;
		case LogicOperation::And         : return VK_LOGIC_OP_AND;
		case LogicOperation::Nand        : return VK_LOGIC_OP_NAND;
		case LogicOperation::Or          : return VK_LOGIC_OP_OR;
		case LogicOperation::Nor         : return VK_LOGIC_OP_NOR;
		case LogicOperation::Xor         : return VK_LOGIC_OP_XOR;
		case LogicOperation::Equiv       : return VK_LOGIC_OP_EQUIVALENT;
		case LogicOperation::AndReverse  : return VK_LOGIC_OP_AND_REVERSE;
		case LogicOperation::AndInverted : return VK_LOGIC_OP_AND_INVERTED;
		case LogicOperation::OrReverse   : return VK_LOGIC_OP_OR_REVERSE;
		case LogicOperation::OrInverted  : return VK_LOGIC_OP_OR_INVERTED;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return {};
	}

	VkBlendOp toVulkan(BlendOperation op)
	{
		// The advanced blend operations require VK_EXT_blend_operation_advanced
		// clang-format off
		switch (op) {
		case BlendOperation::Add          : return VK_BLEND_OP_ADD;
		case BlendOperation::Subtract     : return VK_BLEND_OP_SUBTRACT;
		case BlendOperation::RevSubtract  : return VK_BLEND_OP_REVERSE_SUBTRACT;
		case BlendOperation::Min          : return VK_BLEND_OP_MIN;
		case BlendOperation::Max          : return VK_BLEND_OP_MAX;
		case BlendOperation::Multiply     : return VK_BLEND_OP_MULTIPLY_EXT;
		case BlendOperation::Screen       : return VK_BLEND_OP_SCREEN_EXT;
		case BlendOperation::Overlay      : return VK_BLEND_OP_OVERLAY_EXT;
		case BlendOperation::Darken       : return VK_BLEND_OP_DARKEN_EXT;
		case BlendOperation::Lighten      : return VK_BLEND_OP_LIGHTEN_EXT;
		case BlendOperation::Colordodge   : return VK_BLEND_OP_COLORDODGE_EXT;
		case BlendOperation::Colorburn    : return VK_BLEND_OP_COLORBURN_EXT;
		case BlendOperation::Hardlight    : return VK_BLEND_OP_HARDLIGHT_EXT;
		case BlendOperation::Softlight    : return VK_BLEND_OP_SOFTLIGHT_EXT;
		case BlendOperation::Difference   : return VK_BLEND_OP_DIFFERENCE_EXT;
		case BlendOperation::Exclusion    : return VK_BLEND_OP_EXCLUSION_EXT;
		case BlendOperation::HslHue       : return VK_BLEND_OP_HSL_HUE_EXT;
		case BlendOperation::HslSaturation: return VK_BLEND_OP_HSL_SATURATION_EXT;
		case BlendOperation::HslColor     : return VK_BLEND_OP_HSL_COLOR_EXT;
		case BlendOperation::HslLuminosity: return VK_BLEND_OP_HSL_LUMINOSITY_EXT;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return {};
	}

	VkBlendFactor toVulkan(Blend factor)
	{
		// clang-format off
		switch (factor) {
		case Blend::Zero          : return VK_BLEND_FACTOR_ZERO;
		case Blend::One           : return VK_BLEND_FACTOR_ONE;
		case Blend::SrcColour     : return VK_BLEND_FACTOR_SRC_COLOR;
		case Blend::InvSrcColour  : return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
		case Blend::SrcAlpha      : return VK_BLEND_FACTOR_SRC_ALPHA;
		case Blend::InvSrcAlpha   : return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		case Blend::DestAlpha     : return VK_BLEND_FACTOR_DST_ALPHA;
		case Blend::InvDestAlpha  : return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
		case Blend::DestColour    : return VK_BLEND_FACTOR_DST_COLOR;
		case Blend::InvDestColour : return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
		case Blend::SrcAlphaSat   : return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
		case Blend::BlendFactor   : return VK_BLEND_FACTOR_CONSTANT_COLOR;
		case Blend::InvBlendFactor: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
		case Blend::Src1Colour    : return VK_BLEND_FACTOR_SRC1_COLOR;
		case Blend::InvSrc1Colour : return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
		case Blend::Src1Alpha     : return VK_BLEND_FACTOR_SRC1_ALPHA;
		case Blend::InvSrc1Alpha  : return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return {};
	}

	VkColorComponentFlags toVulkan(Flags<ColourWriteEnable> mask)
	{
		VkColorComponentFlags flags = 0;
		if (mask.isSet(ColourWriteEnable::Red))
			flags |= VK_COLOR_COMPONENT_R_BIT;
		if (mask.isSet(ColourWriteEnable::Green))
			flags |= VK_COLOR_COMPONENT_G_BIT;
		if (mask.isSet(ColourWriteEnable::Blue))
			flags |= VK_COLOR_COMPONENT_B_BIT;
		if (mask.isSet(ColourWriteEnable::Alpha))
			flags |= VK_COLOR_COMPONENT_A_BIT;

		return flags;
	}

	VkPipelineColorBlendAttachmentState toVulkan(const RenderTargetBlendDescription& desc)
	{
		VkPipelineColorBlendAttachmentState vk_desc;
		vk_desc.blendEnable = desc.BlendEnable ? VK_TRUE : VK_FALSE;
		vk_desc.srcColorBlendFactor = toVulkan(desc.SrcBlend);
		vk_desc.dstColorBlendFactor = toVulkan(desc.DestBlend);
		vk_desc.colorBlendOp = toVulkan(desc.BlendOp);
		vk_desc.srcAlphaBlendFactor = toVulkan(desc.SrcBlendAlpha);
		vk_desc.dstAlphaBlendFactor = toVulkan(desc.DestBlendAlpha);
		vk_desc.alphaBlendOp = toVulkan(desc.BlendOpAlpha);
		vk_desc.colorWriteMask = toVulkan(desc.RenderTargetWriteMask);

		return vk_desc;
	}

	VkPipelineColorBlendStateCreateInfo toVulkan(const BlendDescription& desc, stdext::span<VkPipelineColorBlendAttachmentState> attachments)
	{
		VclRequire(attachments.size() <= desc.RenderTarget.size(), "Number of render-targets is supported.");
		VclRequire(implies(desc.LogicOpEnable, std::none_of(desc.RenderTarget.begin(), desc.RenderTarget.end(), [](const auto& rt) { return rt.BlendEnable; })), "Either logic ops or blend ops are enabled.");

		// Without independent blending, all render-targets use the first configuration
		for (size_t i = 0; i < attachments.size(); i++)
		{
			attachments[i] = toVulkan(desc.IndependentBlendEnable ? desc.RenderTarget[i] : desc.RenderTarget[0]);
		}

		VkPipelineColorBlendStateCreateInfo vk_desc = {};
		vk_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		vk_desc.logicOpEnable = desc.LogicOpEnable ? VK_TRUE : VK_FALSE;
		vk_desc.logicOp = toVulkan(desc.LogicOp);
		vk_desc.attachmentCount = static_cast<uint32_t>(attachments.size());
		vk_desc.pAttachments = attachments.data();
		std::copy(desc.ConstantColor.begin(), desc.ConstantColor.end(), vk_desc.blendConstants);

		return vk_desc;
	}
}}}}
