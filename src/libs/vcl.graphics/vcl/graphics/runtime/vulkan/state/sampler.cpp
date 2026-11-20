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
#include <vcl/graphics/runtime/vulkan/state/sampler.h>

// C++ standard library
#include <algorithm>

// VCL
#include <vcl/core/contract.h>
#include <vcl/core/enummap.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {
	namespace {
		// clang-format off
		constexpr EnumMapping<TextureAddressMode, VkSamplerAddressMode> AddressModesTable[] = {
			{ TextureAddressMode::Wrap,       VK_SAMPLER_ADDRESS_MODE_REPEAT },
			{ TextureAddressMode::Mirror,     VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT },
			{ TextureAddressMode::Clamp,      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE },
			{ TextureAddressMode::Border,     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER },
			{ TextureAddressMode::MirrorOnce, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE },
		};
		// clang-format on
		constexpr EnumMap<TextureAddressMode, VkSamplerAddressMode, enumMapSize(AddressModesTable)> AddressModes{ AddressModesTable };

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

		// Bits of 'FilterType', which follows the encoding of D3D12_FILTER
		constexpr unsigned int MipLinearBit = 0x1;
		constexpr unsigned int MagLinearBit = 0x4;
		constexpr unsigned int MinLinearBit = 0x10;
		constexpr unsigned int AnisotropicBit = 0x40;
		constexpr unsigned int ComparisonBit = 0x80;

		//! Vulkan only provides a fixed set of border colours
		VkBorderColor toVulkanBorderColor(const std::array<float, 4>& colour)
		{
			const std::array<float, 4> opaque_black = { 0, 0, 0, 1 };
			const std::array<float, 4> opaque_white = { 1, 1, 1, 1 };

			if (colour == opaque_black)
				return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
			if (colour == opaque_white)
				return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

			VclRequire((colour == std::array<float, 4>{ 0, 0, 0, 0 }), "Border colour is transparent black, opaque black or opaque white.");
			return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
		}
	}

	VkSamplerCreateInfo toVulkan(const SamplerDescription& desc, bool anisotropy_supported)
	{
		const auto filter = static_cast<unsigned int>(desc.Filter);
		const bool anisotropic = (filter & AnisotropicBit) != 0 && anisotropy_supported && desc.MaxAnisotropy > 1;

		VkSamplerCreateInfo vk_desc = {};
		vk_desc.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		vk_desc.magFilter = (filter & MagLinearBit) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
		vk_desc.minFilter = (filter & MinLinearBit) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
		vk_desc.mipmapMode = (filter & MipLinearBit) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
		vk_desc.addressModeU = AddressModes[desc.AddressU];
		vk_desc.addressModeV = AddressModes[desc.AddressV];
		vk_desc.addressModeW = AddressModes[desc.AddressW];
		vk_desc.mipLodBias = desc.MipLODBias;
		vk_desc.anisotropyEnable = anisotropic ? VK_TRUE : VK_FALSE;
		vk_desc.maxAnisotropy = anisotropic ? static_cast<float>(desc.MaxAnisotropy) : 1.0f;
		vk_desc.compareEnable = (filter & ComparisonBit) ? VK_TRUE : VK_FALSE;
		vk_desc.compareOp = CompareOps[desc.ComparisonFunc];

		// The shared description uses the full float range to disable LOD clamping
		vk_desc.minLod = std::max(desc.MinLOD, 0.0f);
		vk_desc.maxLod = std::min(desc.MaxLOD, VK_LOD_CLAMP_NONE);
		vk_desc.borderColor = toVulkanBorderColor(desc.BorderColor);
		vk_desc.unnormalizedCoordinates = VK_FALSE;

		return vk_desc;
	}
}}}}
