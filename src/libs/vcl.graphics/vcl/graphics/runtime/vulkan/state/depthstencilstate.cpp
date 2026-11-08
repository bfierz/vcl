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
#include <vcl/core/enummap.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {

	namespace {
		// clang-format off
		constexpr EnumMapping<ComparisonFunction, VkCompareOp> CompareOpsTable[] = {
			{ ComparisonFunction::Never,        VK_COMPARE_OP_NEVER },
			{ ComparisonFunction::Less,         VK_COMPARE_OP_LESS },
			{ ComparisonFunction::Equal,        VK_COMPARE_OP_EQUAL },
			{ ComparisonFunction::LessEqual,    VK_COMPARE_OP_LESS_OR_EQUAL },
			{ ComparisonFunction::Greater,      VK_COMPARE_OP_GREATER },
			{ ComparisonFunction::NotEqual,     VK_COMPARE_OP_NOT_EQUAL },
			{ ComparisonFunction::GreaterEqual, VK_COMPARE_OP_GREATER_OR_EQUAL },
			{ ComparisonFunction::Always,       VK_COMPARE_OP_ALWAYS },
		};
		// clang-format on
		constexpr EnumMap<ComparisonFunction, VkCompareOp, enumMapSize(CompareOpsTable)> CompareOps{ CompareOpsTable };

		// clang-format off
		constexpr EnumMapping<StencilOperation, VkStencilOp> StencilOpsTable[] = {
			{ StencilOperation::Keep,             VK_STENCIL_OP_KEEP },
			{ StencilOperation::Zero,             VK_STENCIL_OP_ZERO },
			{ StencilOperation::Replace,          VK_STENCIL_OP_REPLACE },
			{ StencilOperation::IncreaseSaturate, VK_STENCIL_OP_INCREMENT_AND_CLAMP },
			{ StencilOperation::DecreaseSaturate, VK_STENCIL_OP_DECREMENT_AND_CLAMP },
			{ StencilOperation::Invert,           VK_STENCIL_OP_INVERT },
			{ StencilOperation::IncreaseWrap,     VK_STENCIL_OP_INCREMENT_AND_WRAP },
			{ StencilOperation::DecreaseWrap,     VK_STENCIL_OP_DECREMENT_AND_WRAP },
		};
		// clang-format on
		constexpr EnumMap<StencilOperation, VkStencilOp, enumMapSize(StencilOpsTable)> StencilOps{ StencilOpsTable };
	}

	VkCompareOp toVulkan(ComparisonFunction func)
	{
		return CompareOps[func];
	}

	VkStencilOp toVulkan(StencilOperation op)
	{
		return StencilOps[op];
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
