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
#include <vcl/core/enummap.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {
	namespace {
		// clang-format off
		constexpr EnumMapping<LogicOperation, VkLogicOp> LogicOpsTable[] = {
			{ LogicOperation::Clear,        VK_LOGIC_OP_CLEAR },
			{ LogicOperation::Set,          VK_LOGIC_OP_SET },
			{ LogicOperation::Copy,         VK_LOGIC_OP_COPY },
			{ LogicOperation::CopyInverted, VK_LOGIC_OP_COPY_INVERTED },
			{ LogicOperation::NoOp,         VK_LOGIC_OP_NO_OP },
			{ LogicOperation::Invert,       VK_LOGIC_OP_INVERT },
			{ LogicOperation::And,          VK_LOGIC_OP_AND },
			{ LogicOperation::Nand,         VK_LOGIC_OP_NAND },
			{ LogicOperation::Or,           VK_LOGIC_OP_OR },
			{ LogicOperation::Nor,          VK_LOGIC_OP_NOR },
			{ LogicOperation::Xor,          VK_LOGIC_OP_XOR },
			{ LogicOperation::Equiv,        VK_LOGIC_OP_EQUIVALENT },
			{ LogicOperation::AndReverse,   VK_LOGIC_OP_AND_REVERSE },
			{ LogicOperation::AndInverted,  VK_LOGIC_OP_AND_INVERTED },
			{ LogicOperation::OrReverse,    VK_LOGIC_OP_OR_REVERSE },
			{ LogicOperation::OrInverted,   VK_LOGIC_OP_OR_INVERTED },
		};
		// clang-format on
		constexpr EnumMap<LogicOperation, VkLogicOp, enumMapSize(LogicOpsTable)> LogicOps{ LogicOpsTable };

		// The advanced blend operations require VK_EXT_blend_operation_advanced
		// clang-format off
		constexpr EnumMapping<BlendOperation, VkBlendOp> BlendOpsTable[] = {
			{ BlendOperation::Add,           VK_BLEND_OP_ADD },
			{ BlendOperation::Subtract,      VK_BLEND_OP_SUBTRACT },
			{ BlendOperation::RevSubtract,   VK_BLEND_OP_REVERSE_SUBTRACT },
			{ BlendOperation::Min,           VK_BLEND_OP_MIN },
			{ BlendOperation::Max,           VK_BLEND_OP_MAX },
			{ BlendOperation::Multiply,      VK_BLEND_OP_MULTIPLY_EXT },
			{ BlendOperation::Screen,        VK_BLEND_OP_SCREEN_EXT },
			{ BlendOperation::Overlay,       VK_BLEND_OP_OVERLAY_EXT },
			{ BlendOperation::Darken,        VK_BLEND_OP_DARKEN_EXT },
			{ BlendOperation::Lighten,       VK_BLEND_OP_LIGHTEN_EXT },
			{ BlendOperation::Colordodge,    VK_BLEND_OP_COLORDODGE_EXT },
			{ BlendOperation::Colorburn,     VK_BLEND_OP_COLORBURN_EXT },
			{ BlendOperation::Hardlight,     VK_BLEND_OP_HARDLIGHT_EXT },
			{ BlendOperation::Softlight,     VK_BLEND_OP_SOFTLIGHT_EXT },
			{ BlendOperation::Difference,    VK_BLEND_OP_DIFFERENCE_EXT },
			{ BlendOperation::Exclusion,     VK_BLEND_OP_EXCLUSION_EXT },
			{ BlendOperation::HslHue,        VK_BLEND_OP_HSL_HUE_EXT },
			{ BlendOperation::HslSaturation, VK_BLEND_OP_HSL_SATURATION_EXT },
			{ BlendOperation::HslColor,      VK_BLEND_OP_HSL_COLOR_EXT },
			{ BlendOperation::HslLuminosity, VK_BLEND_OP_HSL_LUMINOSITY_EXT },
		};
		// clang-format on
		constexpr EnumMap<BlendOperation, VkBlendOp, enumMapSize(BlendOpsTable)> BlendOps{ BlendOpsTable };

		// clang-format off
		constexpr EnumMapping<Blend, VkBlendFactor> BlendFactorsTable[] = {
			{ Blend::Zero,           VK_BLEND_FACTOR_ZERO },
			{ Blend::One,            VK_BLEND_FACTOR_ONE },
			{ Blend::SrcColour,      VK_BLEND_FACTOR_SRC_COLOR },
			{ Blend::InvSrcColour,   VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR },
			{ Blend::SrcAlpha,       VK_BLEND_FACTOR_SRC_ALPHA },
			{ Blend::InvSrcAlpha,    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
			{ Blend::DestAlpha,      VK_BLEND_FACTOR_DST_ALPHA },
			{ Blend::InvDestAlpha,   VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA },
			{ Blend::DestColour,     VK_BLEND_FACTOR_DST_COLOR },
			{ Blend::InvDestColour,  VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR },
			{ Blend::SrcAlphaSat,    VK_BLEND_FACTOR_SRC_ALPHA_SATURATE },
			{ Blend::BlendFactor,    VK_BLEND_FACTOR_CONSTANT_COLOR },
			{ Blend::InvBlendFactor, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR },
			{ Blend::Src1Colour,     VK_BLEND_FACTOR_SRC1_COLOR },
			{ Blend::InvSrc1Colour,  VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR },
			{ Blend::Src1Alpha,      VK_BLEND_FACTOR_SRC1_ALPHA },
			{ Blend::InvSrc1Alpha,   VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA },
		};
		// clang-format on
		constexpr EnumMap<Blend, VkBlendFactor, enumMapSize(BlendFactorsTable)> BlendFactors{ BlendFactorsTable };
	}

	VkLogicOp toVulkan(LogicOperation op)
	{
		return LogicOps[op];
	}

	VkBlendOp toVulkan(BlendOperation op)
	{
		return BlendOps[op];
	}

	VkBlendFactor toVulkan(Blend factor)
	{
		return BlendFactors[factor];
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
