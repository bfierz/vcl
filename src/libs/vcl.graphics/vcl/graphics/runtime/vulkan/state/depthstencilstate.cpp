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
#include <vcl/graphics/runtime/vulkan/state/depthstencilstate.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {

	VkCompareOp toVulkan(ComparisonFunction func)
	{
		// clang-format off
		switch (func) {
		case ComparisonFunction::Never       : return VK_COMPARE_OP_NEVER;
		case ComparisonFunction::Less        : return VK_COMPARE_OP_LESS;
		case ComparisonFunction::Equal       : return VK_COMPARE_OP_EQUAL;
		case ComparisonFunction::LessEqual   : return VK_COMPARE_OP_LESS_OR_EQUAL;
		case ComparisonFunction::Greater     : return VK_COMPARE_OP_GREATER;
		case ComparisonFunction::NotEqual    : return VK_COMPARE_OP_NOT_EQUAL;
		case ComparisonFunction::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
		case ComparisonFunction::Always      : return VK_COMPARE_OP_ALWAYS;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return {};
	}

	VkStencilOp toVulkan(StencilOperation op)
	{
		// clang-format off
		switch (op) {
		case StencilOperation::Keep            : return VK_STENCIL_OP_KEEP;
		case StencilOperation::Zero            : return VK_STENCIL_OP_ZERO;
		case StencilOperation::Replace         : return VK_STENCIL_OP_REPLACE;
		case StencilOperation::IncreaseSaturate: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
		case StencilOperation::DecreaseSaturate: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
		case StencilOperation::Invert          : return VK_STENCIL_OP_INVERT;
		case StencilOperation::IncreaseWrap    : return VK_STENCIL_OP_INCREMENT_AND_WRAP;
		case StencilOperation::DecreaseWrap    : return VK_STENCIL_OP_DECREMENT_AND_WRAP;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return {};
	}

	VkStencilOpState toVulkan(const DepthStencilOperationDescription& desc, uint8_t read_mask, uint8_t write_mask)
	{
		VkStencilOpState vk_desc;
		vk_desc.failOp = toVulkan(desc.StencilFailOp);
		vk_desc.passOp = toVulkan(desc.StencilPassOp);
		vk_desc.depthFailOp = toVulkan(desc.StencilDepthFailOp);
		vk_desc.compareOp = toVulkan(desc.StencilFunc);
		vk_desc.compareMask = read_mask;
		vk_desc.writeMask = write_mask;

		// The reference value is set dynamically
		vk_desc.reference = 0;

		return vk_desc;
	}

	VkPipelineDepthStencilStateCreateInfo toVulkan(const DepthStencilDescription& desc)
	{
		VkPipelineDepthStencilStateCreateInfo vk_desc = {};
		vk_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		vk_desc.depthTestEnable = desc.DepthEnable ? VK_TRUE : VK_FALSE;
		vk_desc.depthWriteEnable = desc.DepthWriteMask == DepthWriteMethod::All ? VK_TRUE : VK_FALSE;
		vk_desc.depthCompareOp = toVulkan(desc.DepthFunc);
		vk_desc.depthBoundsTestEnable = VK_FALSE;
		vk_desc.stencilTestEnable = desc.StencilEnable ? VK_TRUE : VK_FALSE;
		vk_desc.front = toVulkan(desc.FrontFace, desc.StencilReadMask, desc.StencilWriteMask);
		vk_desc.back = toVulkan(desc.BackFace, desc.StencilReadMask, desc.StencilWriteMask);
		vk_desc.minDepthBounds = 0.0f;
		vk_desc.maxDepthBounds = 1.0f;

		return vk_desc;
	}
}}}}
