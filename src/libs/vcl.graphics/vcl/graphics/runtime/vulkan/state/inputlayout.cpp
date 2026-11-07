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
#include <vcl/graphics/runtime/vulkan/state/inputlayout.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/vulkan/tools.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {

	VkVertexInputRate toVulkan(VertexDataClassification classification)
	{
		// clang-format off
		switch (classification) {
		case VertexDataClassification::VertexDataPerObject  : return VK_VERTEX_INPUT_RATE_VERTEX;
		case VertexDataClassification::VertexDataPerInstance: return VK_VERTEX_INPUT_RATE_INSTANCE;
		default: { VclDebugError("Enumeration value is valid."); }
		}
		// clang-format on

		return {};
	}

	VkPipelineVertexInputStateCreateInfo toVulkan(
		const InputLayoutDescription& desc,
		std::vector<VkVertexInputBindingDescription>& bindings,
		std::vector<VkVertexInputAttributeDescription>& attributes)
	{
		using Vcl::Graphics::Vulkan::convert;

		bindings.clear();
		bindings.reserve(desc.bindings().size());
		for (const auto& binding : desc.bindings())
		{
			bindings.push_back({ binding.Binding, binding.Stride, toVulkan(binding.InputRate) });
		}

		// Attributes spanning multiple locations (e.g., matrices) occupy one location per column
		attributes.clear();
		attributes.reserve(desc.attributes().size());
		for (size_t idx = 0; idx < desc.attributes().size(); idx++)
		{
			const auto& attrib = desc.attributes()[idx];
			const auto format = convert(attrib.Format);
			const uint32_t nr_locations = std::max(1u, attrib.NumberLocations);
			const uint32_t location_size = Vcl::Graphics::sizeInBytes(attrib.Format);
			for (uint32_t l = 0; l < nr_locations; l++)
			{
				attributes.push_back({ static_cast<uint32_t>(desc.location(idx)) + l, attrib.InputSlot, format, attrib.Offset + l * location_size });
			}
		}

		VkPipelineVertexInputStateCreateInfo vi = {};
		vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vi.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
		vi.pVertexBindingDescriptions = bindings.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
		vi.pVertexAttributeDescriptions = attributes.data();

		return vi;
	}
}}}}
