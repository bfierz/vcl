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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <memory>

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/graphics/runtime/resource/texture.h>
#include <vcl/graphics/runtime/vulkan/resource/buffer.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/memory.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {
	struct GenericTextureDescription : public TextureBaseDescription
	{
		GenericTextureDescription(TextureType type, const Texture1DDescription& desc);
		GenericTextureDescription(TextureType type, const Texture2DDescription& desc);
		GenericTextureDescription(TextureType type, const Texture3DDescription& desc);
		GenericTextureDescription(TextureType type, const TextureCubeDescription& desc);

		TextureType Type;
		uint32_t Width;
		uint32_t Height;
		uint32_t Depth;
		uint32_t MipLevels;
		uint32_t ArraySize;
	};

	class Texture : public Runtime::Texture
	{
	public:
		//! Create a new texture
		//! \param context Vulkan context owning the texture
		//! \param desc Description of the texture
		//! \param init_data Optional initial data of the first mip-map level
		//! \param cmd_buffer Command buffer recording the upload of \p init_data.
		//!        The upload is complete once \p cmd_buffer was executed.
		Texture(Vcl::Graphics::Vulkan::Context* context, const GenericTextureDescription& desc, const TextureResource* init_data = nullptr, VkCommandBuffer cmd_buffer = VK_NULL_HANDLE);
		Texture(const Texture&) = delete;
		~Texture() override;

		//! Convert to Vulkan ID
		VkImage id() const { return _image; }

		//! Access the view on the entire texture
		VkImageView view() const { return _view; }

		//! Access the layout the texture is in after all recorded commands are executed
		VkImageLayout layout() const { return _layout; }

		//! Access the aspects of the texture's format
		VkImageAspectFlags aspects() const { return _aspects; }

		//! Record a layout transition of the entire texture
		//! \param cmd_buffer Command buffer to record the transition
		//! \param layout Target layout
		void transition(VkCommandBuffer cmd_buffer, VkImageLayout layout);

		//! Record a copy of the first mip-map level of all layers to a buffer
		void copyTo(VkCommandBuffer cmd_buffer, Buffer& target);

	private:
		//! Owning context
		Vcl::Graphics::Vulkan::Context* _context{ nullptr };

		//! Vulkan image
		VkImage _image{ VK_NULL_HANDLE };

		//! View on the entire image
		VkImageView _view{ VK_NULL_HANDLE };

		//! Memory backing the image
		std::unique_ptr<Vcl::Graphics::Vulkan::Memory> _memory;

		//! Staging buffer used to upload the initial data
		std::unique_ptr<Buffer> _uploadBuffer;

		//! Current layout of the image
		VkImageLayout _layout{ VK_IMAGE_LAYOUT_UNDEFINED };

		//! Aspects of the image format
		VkImageAspectFlags _aspects{ 0 };
	};

#define VCL_VULKAN_DECLARE_TEXTURE(name, description)                                                                                                                    \
	class name final : public Texture                                                                                                                                    \
	{                                                                                                                                                                    \
	public:                                                                                                                                                              \
		name(Vcl::Graphics::Vulkan::Context* context, const description& desc, const TextureResource* init_data = nullptr, VkCommandBuffer cmd_buffer = VK_NULL_HANDLE); \
		std::unique_ptr<Runtime::Texture> clone() const override;                                                                                                        \
	}

	VCL_VULKAN_DECLARE_TEXTURE(Texture1D, Texture1DDescription);
	VCL_VULKAN_DECLARE_TEXTURE(Texture1DArray, Texture1DDescription);
	VCL_VULKAN_DECLARE_TEXTURE(Texture2D, Texture2DDescription);
	VCL_VULKAN_DECLARE_TEXTURE(Texture2DArray, Texture2DDescription);
	VCL_VULKAN_DECLARE_TEXTURE(Texture3D, Texture3DDescription);
	VCL_VULKAN_DECLARE_TEXTURE(TextureCube, TextureCubeDescription);
	VCL_VULKAN_DECLARE_TEXTURE(TextureCubeArray, TextureCubeDescription);

#undef VCL_VULKAN_DECLARE_TEXTURE
}}}}
