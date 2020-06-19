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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

// VCL
#include <vcl/graphics/d3d12/device.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/resource/texture.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12
{
	struct GenericTextureDescription : public TextureBaseDescription
	{
		GenericTextureDescription(const Texture1DDescription& desc);
		GenericTextureDescription(const Texture2DDescription& desc);
		GenericTextureDescription(const Texture3DDescription& desc);
		GenericTextureDescription(const TextureCubeDescription& desc);

		TextureType Type;
		uint32_t Width;
		uint32_t Height;
		uint32_t Depth;
		uint32_t MipLevels;
		uint32_t ArraySize;
	};

	class Texture : public Runtime::Texture, public Resource
	{
	public:
		Texture(Graphics::D3D12::Device* device, const GenericTextureDescription& desc, const TextureResource* init_data = nullptr, ID3D12GraphicsCommandList* cmd_queue = nullptr);
		Texture(const Texture&);

		virtual D3D12_SHADER_RESOURCE_VIEW_DESC srv() const;
		virtual D3D12_UNORDERED_ACCESS_VIEW_DESC uav() const;

		void copyTo(ID3D12GraphicsCommandList* cmd_queue, Buffer& target);
		void copyTo(ID3D12GraphicsCommandList* cmd_queue, Texture& target);

	private:
		//! 
		ComPtr<ID3D12Resource> _uploadResource;
	};

	class Texture1D final : public Texture
	{
	public:
		Texture1D(Graphics::D3D12::Device* device, const Texture1DDescription& desc, const TextureResource* init_data = nullptr, ID3D12GraphicsCommandList* cmd_queue = nullptr);
		Texture1D(const Texture1D&);

		std::unique_ptr<Runtime::Texture> clone() const override;
	};

	class Texture1DArray final : public Texture
	{
	public:
		Texture1DArray(Graphics::D3D12::Device* device, const Texture1DDescription& desc, const TextureResource* init_data = nullptr, ID3D12GraphicsCommandList* cmd_queue = nullptr);
		Texture1DArray(const Texture1DArray&);

		std::unique_ptr<Runtime::Texture> clone() const override;
	};

	class Texture2D final : public Texture
	{
	public:
		Texture2D(Graphics::D3D12::Device* device, const Texture2DDescription& desc, const TextureResource* init_data = nullptr, ID3D12GraphicsCommandList* cmd_queue = nullptr);
		Texture2D(const Texture2D&);

		std::unique_ptr<Runtime::Texture> clone() const override;

		D3D12_SHADER_RESOURCE_VIEW_DESC srv() const override;
		D3D12_UNORDERED_ACCESS_VIEW_DESC uav() const override;

	};

	class Texture2DArray final : public Texture
	{
	public:
		Texture2DArray(Graphics::D3D12::Device* device, const Texture2DDescription& desc, const TextureResource* init_data = nullptr, ID3D12GraphicsCommandList* cmd_queue = nullptr);
		Texture2DArray(const Texture2DArray&);

		std::unique_ptr<Runtime::Texture> clone() const override;
	};

	class Texture3D final : public Texture
	{
	public:
		Texture3D(Graphics::D3D12::Device* device, const Texture3DDescription& desc, const TextureResource* init_data = nullptr, ID3D12GraphicsCommandList* cmd_queue = nullptr);
		Texture3D(const Texture3D&);

		std::unique_ptr<Runtime::Texture> clone() const override;
	};

	class TextureCube final : public Texture
	{
	public:
		TextureCube(Graphics::D3D12::Device* device, const TextureCubeDescription& desc, const TextureResource* init_data = nullptr, ID3D12GraphicsCommandList* cmd_queue = nullptr);
		TextureCube(const TextureCube&);

		std::unique_ptr<Runtime::Texture> clone() const override;
	};

	class TextureCubeArray final : public Texture
	{
	public:
		TextureCubeArray(Graphics::D3D12::Device* device, const TextureCubeDescription& desc, const TextureResource* init_data = nullptr, ID3D12GraphicsCommandList* cmd_queue = nullptr);
		TextureCubeArray(const TextureCubeArray&);

		std::unique_ptr<Runtime::Texture> clone() const override;
	};
}}}}
