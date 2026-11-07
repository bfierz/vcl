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
#include <vcl/graphics/runtime/vulkan/resource/texture.h>

// C++ standard library
#include <algorithm>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/tools.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {
	GenericTextureDescription::GenericTextureDescription(TextureType type, const Texture1DDescription& desc)
	{
		Type = type;
		Format = desc.Format;
		Usage = desc.Usage;
		Width = desc.Width;
		Height = 1;
		Depth = 1;
		ArraySize = std::max(1u, desc.ArraySize);
		MipLevels = std::max(1u, desc.MipLevels);
	}

	GenericTextureDescription::GenericTextureDescription(TextureType type, const Texture2DDescription& desc)
	{
		Type = type;
		Format = desc.Format;
		Usage = desc.Usage;
		Width = desc.Width;
		Height = desc.Height;
		Depth = 1;
		ArraySize = std::max(1u, desc.ArraySize);
		MipLevels = std::max(1u, desc.MipLevels);
	}

	GenericTextureDescription::GenericTextureDescription(TextureType type, const Texture3DDescription& desc)
	{
		Type = type;
		Format = desc.Format;
		Usage = desc.Usage;
		Width = desc.Width;
		Height = desc.Height;
		Depth = desc.Depth;
		ArraySize = 1;
		MipLevels = std::max(1u, desc.MipLevels);
	}

	GenericTextureDescription::GenericTextureDescription(TextureType type, const TextureCubeDescription& desc)
	{
		Type = type;
		Format = desc.Format;
		Usage = desc.Usage;
		Width = desc.Width;
		Height = desc.Height;
		Depth = 1;
		ArraySize = 6 * std::max(1u, desc.ArraySize);
		MipLevels = std::max(1u, desc.MipLevels);
	}

	namespace {
		bool isCube(TextureType type)
		{
			return type == TextureType::TextureCube || type == TextureType::TextureCubeArray;
		}

		VkImageType toVulkanImageType(TextureType type)
		{
			switch (type)
			{
			case TextureType::Texture1D:
			case TextureType::Texture1DArray:
				return VK_IMAGE_TYPE_1D;
			case TextureType::Texture3D:
				return VK_IMAGE_TYPE_3D;
			default:
				return VK_IMAGE_TYPE_2D;
			}
		}

		VkImageViewType toVulkanViewType(TextureType type)
		{
			// clang-format off
			switch (type) {
			case TextureType::Texture1D       : return VK_IMAGE_VIEW_TYPE_1D;
			case TextureType::Texture1DArray  : return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
			case TextureType::Texture2D       : return VK_IMAGE_VIEW_TYPE_2D;
			case TextureType::Texture2DArray  : return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
			case TextureType::Texture3D       : return VK_IMAGE_VIEW_TYPE_3D;
			case TextureType::TextureCube     : return VK_IMAGE_VIEW_TYPE_CUBE;
			case TextureType::TextureCubeArray: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
			default: { VclDebugError("Texture type is supported."); }
			}
			// clang-format on

			return {};
		}

		VkImageAspectFlags toVulkanAspects(SurfaceFormat format)
		{
			switch (format)
			{
			case SurfaceFormat::D16_UNORM:
			case SurfaceFormat::D32_FLOAT:
				return VK_IMAGE_ASPECT_DEPTH_BIT;
			case SurfaceFormat::D24_UNORM_S8_UINT:
			case SurfaceFormat::D32_FLOAT_S8X24_UINT:
				return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
			default:
				return VK_IMAGE_ASPECT_COLOR_BIT;
			}
		}

		VkImageUsageFlags toVulkanUsage(Flags<TextureUsage> usage, VkImageAspectFlags aspects)
		{
			VkImageUsageFlags flags = 0;
			if (usage.isSet(TextureUsage::CopySrc))
				flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
			if (usage.isSet(TextureUsage::CopyDst))
				flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			if (usage.isSet(TextureUsage::Sampled))
				flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
			if (usage.isSet(TextureUsage::Storage))
				flags |= VK_IMAGE_USAGE_STORAGE_BIT;
			if (usage.isSet(TextureUsage::OutputAttachment))
				flags |= (aspects & VK_IMAGE_ASPECT_COLOR_BIT) ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

			return flags;
		}

		//! Pipeline stages and memory accesses using an image in a given layout
		struct LayoutUsage
		{
			VkPipelineStageFlags2 Stages;
			VkAccessFlags2 Access;
		};

		LayoutUsage usageOf(VkImageLayout layout)
		{
			switch (layout)
			{
			case VK_IMAGE_LAYOUT_UNDEFINED:
				return { VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE };
			case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
				return { VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT };
			case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
				return { VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT };
			case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
				return { VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT };
			case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
				return { VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT };
			case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
				return { VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT };
			default:
				return { VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT };
			}
		}
	}

	Texture::Texture(
		Vcl::Graphics::Vulkan::Context* context,
		const GenericTextureDescription& desc,
		const TextureResource* init_data,
		VkCommandBuffer cmd_buffer)
	: _context(context)
	{
		using Vcl::Graphics::Vulkan::convert;
		using Vcl::Graphics::Vulkan::MemoryHeapType;

		VclRequire(context, "Context is set");
		VclRequire(implies(init_data, cmd_buffer != VK_NULL_HANDLE), "Uploading initial data requires a command buffer.");
		VclRequire(implies(init_data, init_data && init_data->verify()), "Initial data is consistent.");

		initializeView(
			desc.Type, desc.Format, desc.Usage,
			0, desc.MipLevels,
			0, desc.ArraySize,
			desc.Width, desc.Height, desc.Depth);

		_aspects = toVulkanAspects(desc.Format);

		VkImageCreateInfo image_info = {};
		image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image_info.flags = isCube(desc.Type) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
		image_info.imageType = toVulkanImageType(desc.Type);
		image_info.format = convert(desc.Format);
		image_info.extent = { desc.Width, desc.Height, desc.Depth };
		image_info.mipLevels = desc.MipLevels;
		image_info.arrayLayers = desc.ArraySize;
		image_info.samples = VK_SAMPLE_COUNT_1_BIT;
		image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
		image_info.usage = toVulkanUsage(usage(), _aspects) | (init_data ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0);
		image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VkResult res = vkCreateImage(*context, &image_info, nullptr, &_image);
		VclCheck(res == VK_SUCCESS, "Image was created.");

		VkMemoryRequirements reqs = {};
		vkGetImageMemoryRequirements(*context, _image, &reqs);
		const auto* heap = context->device()->findMemoryHeap(reqs, MemoryHeapType::Default);
		VclCheck(heap, "Suitable heap found.");
		_memory = std::make_unique<Vcl::Graphics::Vulkan::Memory>(context, heap, reqs.size);

		res = vkBindImageMemory(*context, _image, *_memory, 0);
		VclCheck(res == VK_SUCCESS, "Image was bound to memory.");

		VkImageViewCreateInfo view_info = {};
		view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_info.image = _image;
		view_info.viewType = toVulkanViewType(desc.Type);
		view_info.format = image_info.format;
		view_info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
		view_info.subresourceRange = { _aspects, 0, desc.MipLevels, 0, desc.ArraySize };

		res = vkCreateImageView(*context, &view_info, nullptr, &_view);
		VclCheck(res == VK_SUCCESS, "Image view was created.");

		if (init_data)
		{
			// Stage the data of the first mip-map level of all layers
			const BufferDescription staging_desc = {
				static_cast<uint32_t>(init_data->Data.size()),
				BufferUsage::MapWrite | BufferUsage::CopySrc
			};
			const BufferInitData staging_data = { init_data->data(), init_data->Data.size() };
			_uploadBuffer = std::make_unique<Buffer>(context, staging_desc, &staging_data);

			transition(cmd_buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

			VkBufferImageCopy region = {};
			region.imageSubresource = { _aspects, 0, 0, desc.ArraySize };
			region.imageExtent = { desc.Width, desc.Height, desc.Depth };
			vkCmdCopyBufferToImage(cmd_buffer, _uploadBuffer->id(), _image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		}
	}

	Texture::~Texture()
	{
		if (_view)
			vkDestroyImageView(*_context, _view, nullptr);
		if (_image)
			vkDestroyImage(*_context, _image, nullptr);
	}

	void Texture::transition(VkCommandBuffer cmd_buffer, VkImageLayout layout)
	{
		if (layout == _layout)
			return;

		const auto src = usageOf(_layout);
		const auto dst = usageOf(layout);

		VkImageMemoryBarrier2 barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask = src.Stages;
		barrier.srcAccessMask = src.Access;
		barrier.dstStageMask = dst.Stages;
		barrier.dstAccessMask = dst.Access;
		barrier.oldLayout = _layout;
		barrier.newLayout = layout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = _image;
		barrier.subresourceRange = { _aspects, 0, static_cast<uint32_t>(mipMapLevels()), 0, static_cast<uint32_t>(layers()) };

		VkDependencyInfo dependency = {};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers = &barrier;
		vkCmdPipelineBarrier2(cmd_buffer, &dependency);

		_layout = layout;
	}

	void Texture::copyTo(VkCommandBuffer cmd_buffer, Buffer& target)
	{
		VclRequire(target.sizeInBytes() >= static_cast<size_t>(sizeInBytes() * layers()), "Target buffer is large enough.");

		transition(cmd_buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

		// Copies of depth-stencil formats are restricted to a single aspect
		const VkImageAspectFlags aspect = (_aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : _aspects;

		VkBufferImageCopy region = {};
		region.imageSubresource = { aspect, 0, 0, static_cast<uint32_t>(layers()) };
		region.imageExtent = { static_cast<uint32_t>(width()), static_cast<uint32_t>(height()), static_cast<uint32_t>(depth()) };
		vkCmdCopyImageToBuffer(cmd_buffer, _image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target.id(), 1, &region);
	}

#define VCL_VULKAN_DEFINE_TEXTURE(name, description)                                                                                           \
	name::name(Vcl::Graphics::Vulkan::Context* context, const description& desc, const TextureResource* init_data, VkCommandBuffer cmd_buffer) \
	: Texture(context, GenericTextureDescription{ TextureType::name, desc }, init_data, cmd_buffer)                                            \
	{                                                                                                                                          \
	}                                                                                                                                          \
	std::unique_ptr<Runtime::Texture> name::clone() const                                                                                      \
	{                                                                                                                                          \
		VclDebugError("Vulkan textures cannot be cloned.");                                                                                    \
		return nullptr;                                                                                                                        \
	}

	VCL_VULKAN_DEFINE_TEXTURE(Texture1D, Texture1DDescription)
	VCL_VULKAN_DEFINE_TEXTURE(Texture1DArray, Texture1DDescription)
	VCL_VULKAN_DEFINE_TEXTURE(Texture2D, Texture2DDescription)
	VCL_VULKAN_DEFINE_TEXTURE(Texture2DArray, Texture2DDescription)
	VCL_VULKAN_DEFINE_TEXTURE(Texture3D, Texture3DDescription)
	VCL_VULKAN_DEFINE_TEXTURE(TextureCube, TextureCubeDescription)
	VCL_VULKAN_DEFINE_TEXTURE(TextureCubeArray, TextureCubeDescription)

#undef VCL_VULKAN_DEFINE_TEXTURE
}}}}
