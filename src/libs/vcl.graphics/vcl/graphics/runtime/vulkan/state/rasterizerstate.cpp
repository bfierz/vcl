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
#include <vcl/graphics/runtime/vulkan/state/rasterizerstate.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {

	VkPolygonMode toVulkan(FillModeMethod mode)
	{
		// clang-format off
		switch (mode) {
		case FillModeMethod::Solid    : return VK_POLYGON_MODE_FILL;
		case FillModeMethod::Wireframe: return VK_POLYGON_MODE_LINE;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return {};
	}

	VkCullModeFlags toVulkan(CullModeMethod mode)
	{
		// clang-format off
		switch (mode) {
		case CullModeMethod::None : return VK_CULL_MODE_NONE;
		case CullModeMethod::Front: return VK_CULL_MODE_FRONT_BIT;
		case CullModeMethod::Back : return VK_CULL_MODE_BACK_BIT;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return {};
	}

	VkPipelineRasterizationStateCreateInfo toVulkan(const RasterizerDescription& desc)
	{
		// Disabling depth clipping requires the 'depthClamp' feature, which is not enabled
		VclRequire(desc.DepthClipEnable, "Depth clipping is enabled.");

		VkPipelineRasterizationStateCreateInfo vk_desc = {};
		vk_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		vk_desc.depthClampEnable = VK_FALSE;
		vk_desc.rasterizerDiscardEnable = VK_FALSE;
		vk_desc.polygonMode = toVulkan(desc.FillMode);
		vk_desc.cullMode = toVulkan(desc.CullMode);
		vk_desc.frontFace = desc.FrontCounterClockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
		vk_desc.depthBiasEnable = (desc.DepthBias != 0 || desc.SlopeScaledDepthBias != 0.0f) ? VK_TRUE : VK_FALSE;
		vk_desc.depthBiasConstantFactor = static_cast<float>(desc.DepthBias);
		vk_desc.depthBiasClamp = desc.DepthBiasClamp;
		vk_desc.depthBiasSlopeFactor = desc.SlopeScaledDepthBias;
		vk_desc.lineWidth = 1.0f;

		// The scissor test is always active in Vulkan. 'MultisampleEnable' and
		// 'AntialiasedLineEnable' require VK_EXT_line_rasterization.

		return vk_desc;
	}
}}}}
