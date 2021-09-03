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
#include <vcl/graphics/runtime/d3d12/resource/texture.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/3rdparty/d3dx12.h>
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/math/ceil.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 {
	GenericTextureDescription::GenericTextureDescription(const Texture1DDescription& desc)
	{
		Type = TextureType::Texture1D;
		Format = desc.Format;
		Usage = desc.Usage;
		Width = desc.Width;
		Height = 1;
		Depth = 1;
		ArraySize = desc.ArraySize;
		MipLevels = desc.MipLevels;
	}
	GenericTextureDescription::GenericTextureDescription(const Texture2DDescription& desc)
	{
		Type = TextureType::Texture2D;
		Format = desc.Format;
		Usage = desc.Usage;
		Width = desc.Width;
		Height = desc.Height;
		Depth = 1;
		ArraySize = desc.ArraySize;
		MipLevels = desc.MipLevels;
	}
	GenericTextureDescription::GenericTextureDescription(const Texture3DDescription& desc)
	{
		Type = TextureType::Texture3D;
		Format = desc.Format;
		Usage = desc.Usage;
		Width = desc.Width;
		Height = desc.Height;
		Depth = desc.Depth;
		ArraySize = 1;
		MipLevels = desc.MipLevels;
	}
	GenericTextureDescription::GenericTextureDescription(const TextureCubeDescription& desc)
	{
		Type = TextureType::TextureCube;
		Format = desc.Format;
		Usage = desc.Usage;
		Width = desc.Width;
		Height = desc.Height;
		Depth = 1;
		ArraySize = 6 * desc.ArraySize;
		MipLevels = desc.MipLevels;
	}

	D3D12_RESOURCE_STATES toD3DResourceState(Flags<TextureUsage> flag)
	{
		UINT d3d_flags = 0;
		d3d_flags |= (flag.isSet(TextureUsage::CopySrc)) ? D3D12_RESOURCE_STATE_COPY_SOURCE : 0;
		d3d_flags |= (flag.isSet(TextureUsage::CopyDst)) ? D3D12_RESOURCE_STATE_COPY_DEST : 0;
		d3d_flags |= (flag.isSet(TextureUsage::Sampled)) ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE : 0;
		d3d_flags |= (flag.isSet(TextureUsage::Storage)) ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : 0;
		d3d_flags |= (flag.isSet(TextureUsage::OutputAttachment)) ? D3D12_RESOURCE_STATE_RENDER_TARGET : 0;

		return (D3D12_RESOURCE_STATES)d3d_flags;
	}

	D3D12_RESOURCE_DIMENSION toD3DResourceDimension(TextureType type)
	{
		switch (type)
		{
		case TextureType::Texture1D:
		case TextureType::Texture1DArray:
			return D3D12_RESOURCE_DIMENSION_TEXTURE1D;
		case TextureType::Texture2D:
		case TextureType::Texture2DArray:
		case TextureType::Texture2DMS:
		case TextureType::Texture2DMSArray:
		case TextureType::TextureCube:
		case TextureType::TextureCubeArray:
			return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		case TextureType::Texture3D:
			return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
		}

		return D3D12_RESOURCE_DIMENSION_UNKNOWN;
	}

	Texture::Texture(
		Graphics::D3D12::Device* device,
		const GenericTextureDescription& desc,
		const TextureResource* init_data,
		ID3D12GraphicsCommandList* cmd_queue)
	{
		initializeView(
			desc.Type, desc.Format, desc.Usage,
			0, desc.MipLevels,
			0, desc.ArraySize,
			desc.Width, desc.Height, desc.Depth);

		D3D12_RESOURCE_STATES texture_usage;
		texture_usage = D3D12_RESOURCE_STATE_COMMON;
		_targetStates = toD3DResourceState(usage());
		_currentStates = init_data ? D3D12_RESOURCE_STATE_COPY_DEST : texture_usage;

		D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
		if (usage().isSet(TextureUsage::Storage))
			flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

		const D3D12_RESOURCE_DESC d3d12_desc = {
			toD3DResourceDimension(type()),
			0,
			width(),
			height(),
			std::max(depth(), layers()),
			mipMapLevels(),
			Graphics::D3D12::D3D::toD3Denum(desc.Format),
			{ 1, 0 },
			D3D12_TEXTURE_LAYOUT_UNKNOWN,
			flags
		};

		ID3D12Device* d3d12_dev = device->nativeDevice();
		const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		VCL_DIRECT3D_SAFE_CALL(d3d12_dev->CreateCommittedResource(
			&heap_props,
			D3D12_HEAP_FLAG_NONE,
			&d3d12_desc,
			_currentStates,
			nullptr,
			IID_PPV_ARGS(&_resource)));

		uint64_t textureMemorySize = 0;
		//UINT numRows[1];
		//UINT64 rowSizesInBytes[1];
		//D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[1];
		d3d12_dev->GetCopyableFootprints(&d3d12_desc, 0, layers(), 0, nullptr, nullptr, nullptr, &textureMemorySize);

		if (init_data)
		{
			const auto upload_heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
			const auto upload_buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(std::max(init_data->Data.size(), textureMemorySize));
			VCL_DIRECT3D_SAFE_CALL(d3d12_dev->CreateCommittedResource(
				&upload_heap_props,
				D3D12_HEAP_FLAG_NONE,
				&upload_buffer_desc,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&_uploadResource)));

			std::vector<D3D12_SUBRESOURCE_DATA> subresource_data;
			subresource_data.reserve(layers());
			for (int i = 0; i < layers(); i++)
			{
				D3D12_SUBRESOURCE_DATA data = {
					init_data->Data.data() + i * init_data->slicePitch(),
					init_data->rowPitch(),
					init_data->slicePitch()
				};
				subresource_data.emplace_back(data);
			}

			UpdateSubresources(cmd_queue, _resource.Get(), _uploadResource.Get(), 0, 0, layers(), subresource_data.data());
		}
	}
	Texture::Texture(const Texture& rhs)
	: Runtime::Texture(rhs)
	{
	}

	D3D12_SHADER_RESOURCE_VIEW_DESC Texture::srv() const
	{
		return {};
	}
	D3D12_UNORDERED_ACCESS_VIEW_DESC Texture::uav() const
	{
		return {};
	}

	void Texture::copyTo(ID3D12GraphicsCommandList* cmd_queue, Buffer& target)
	{
		using Vcl::Mathematics::ceil;

		VclRequire(usage().isSet(TextureUsage::CopySrc), "Source is copy source");
		VclRequire(target.usage().isSet(BufferUsage::CopyDst), "'target' is copy destination");

		transition(cmd_queue, D3D12_RESOURCE_STATE_COPY_SOURCE);
		target.transition(cmd_queue, D3D12_RESOURCE_STATE_COPY_DEST);

		uint64_t num_subresources = layers() * mipMapLevels();
		uint64_t descr_size = static_cast<uint64_t>(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT64) + sizeof(UINT)) * num_subresources;
		auto descr_mem = std::make_unique<uint8_t[]>(descr_size);

		uint64_t total_size = 0;
		auto layouts = stdext::make_span(reinterpret_cast<D3D12_PLACED_SUBRESOURCE_FOOTPRINT*>(descr_mem.get()), num_subresources);
		auto row_sizes = stdext::make_span(reinterpret_cast<UINT64*>(layouts.data() + num_subresources), num_subresources);
		auto num_rows = stdext::make_span(reinterpret_cast<UINT*>(row_sizes.data() + num_subresources), num_subresources);

		const auto desc = handle()->GetDesc();
		ID3D12Device* d3d12_dev;
		handle()->GetDevice(__uuidof(*d3d12_dev), reinterpret_cast<void**>(&d3d12_dev));
		d3d12_dev->GetCopyableFootprints(&desc, 0, num_subresources, 0, layouts.data(), num_rows.data(), row_sizes.data(), &total_size);
		d3d12_dev->Release();

		for (uint64_t i = 0; i < num_subresources; i++)
		{
			const auto& layout = layouts[i];

			D3D12_TEXTURE_COPY_LOCATION source;
			source.pResource = handle();
			source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			source.SubresourceIndex = i;

			D3D12_TEXTURE_COPY_LOCATION dest;
			dest.pResource = target.handle();
			dest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
			dest.PlacedFootprint = layout;

			D3D12_BOX source_region{ 0, 0, 0, layout.Footprint.Width, layout.Footprint.Height, layout.Footprint.Depth };
			cmd_queue->CopyTextureRegion(&dest, 0, 0, 0, &source, &source_region);
		}
	}

	void Texture::copyTo(ID3D12GraphicsCommandList* cmd_queue, Texture& target)
	{
		using Vcl::Mathematics::ceil;

		VclRequire(usage().isSet(TextureUsage::CopySrc), "Source is copy source");
		VclRequire(target.usage().isSet(TextureUsage::CopyDst), "'target' is copy destination");

		transition(cmd_queue, D3D12_RESOURCE_STATE_COPY_SOURCE);
		target.transition(cmd_queue, D3D12_RESOURCE_STATE_COPY_DEST);

		D3D12_TEXTURE_COPY_LOCATION source;
		source.pResource = handle();
		source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		source.SubresourceIndex = 0;

		D3D12_TEXTURE_COPY_LOCATION dest;
		dest.pResource = target.handle();
		dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		dest.SubresourceIndex = 0;

		D3D12_BOX source_region{ 0, 0, 0, width(), height(), depth() };
		cmd_queue->CopyTextureRegion(&dest, 0, 0, 0, &source, &source_region);
	}

	Texture1D::Texture1D(
		Graphics::D3D12::Device* device,
		const Texture1DDescription& desc,
		const TextureResource* init_data,
		ID3D12GraphicsCommandList* cmd_queue)
	: Texture(device, desc, init_data, cmd_queue)
	{
	}

	Texture1D::Texture1D(const Texture1D& rhs)
	: Texture(rhs)
	{
	}

	std::unique_ptr<Runtime::Texture> Texture1D::clone() const
	{
		return std::make_unique<Texture1D>(*this);
	}

	Texture1DArray::Texture1DArray(
		Graphics::D3D12::Device* device,
		const Texture1DDescription& desc,
		const TextureResource* init_data,
		ID3D12GraphicsCommandList* cmd_queue)
	: Texture(device, desc, init_data, cmd_queue)
	{
	}

	Texture1DArray::Texture1DArray(const Texture1DArray& rhs)
	: Texture(rhs)
	{
	}

	std::unique_ptr<Runtime::Texture> Texture1DArray::clone() const
	{
		return std::make_unique<Texture1DArray>(*this);
	}

	Texture2D::Texture2D(
		Graphics::D3D12::Device* device,
		const Texture2DDescription& desc,
		const TextureResource* init_data,
		ID3D12GraphicsCommandList* cmd_queue)
	: Texture(device, desc, init_data, cmd_queue)
	{
	}

	Texture2D::Texture2D(const Texture2D& rhs)
	: Texture(rhs)
	{
	}

	std::unique_ptr<Runtime::Texture> Texture2D::clone() const
	{
		return std::make_unique<Texture2D>(*this);
	}

	D3D12_SHADER_RESOURCE_VIEW_DESC Texture2D::srv() const
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
		desc.Format = Graphics::D3D12::D3D::toD3Denum(format());
		desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		desc.Texture2D.MostDetailedMip = 0;
		desc.Texture2D.MipLevels = mipMapLevels();
		desc.Texture2D.PlaneSlice = 0;
		desc.Texture2D.ResourceMinLODClamp = 0;
		return desc;
	}
	D3D12_UNORDERED_ACCESS_VIEW_DESC Texture2D::uav() const
	{
		D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
		desc.Format = Graphics::D3D12::D3D::toD3Denum(format());
		desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		desc.Texture2D.MipSlice = 0;
		desc.Texture2D.PlaneSlice = 0;
		return desc;
	}

	Texture2DArray::Texture2DArray(
		Graphics::D3D12::Device* device,
		const Texture2DDescription& desc,
		const TextureResource* init_data,
		ID3D12GraphicsCommandList* cmd_queue)
	: Texture(device, desc, init_data, cmd_queue)
	{
	}

	Texture2DArray::Texture2DArray(const Texture2DArray& rhs)
	: Texture(rhs)
	{
	}

	std::unique_ptr<Runtime::Texture> Texture2DArray::clone() const
	{
		return std::make_unique<Texture2DArray>(*this);
	}

	Texture3D::Texture3D(
		Graphics::D3D12::Device* device,
		const Texture3DDescription& desc,
		const TextureResource* init_data,
		ID3D12GraphicsCommandList* cmd_queue)
	: Texture(device, desc, init_data, cmd_queue)
	{
	}

	Texture3D::Texture3D(const Texture3D& rhs)
	: Texture(rhs)
	{
	}

	std::unique_ptr<Runtime::Texture> Texture3D::clone() const
	{
		return std::make_unique<Texture3D>(*this);
	}

	TextureCube::TextureCube(
		Graphics::D3D12::Device* device,
		const TextureCubeDescription& desc,
		const TextureResource* init_data,
		ID3D12GraphicsCommandList* cmd_queue)
	: Texture(device, desc, init_data, cmd_queue)
	{
	}

	TextureCube::TextureCube(const TextureCube& rhs)
	: Texture(rhs)
	{
	}

	std::unique_ptr<Runtime::Texture> TextureCube::clone() const
	{
		return std::make_unique<TextureCube>(*this);
	}

	TextureCubeArray::TextureCubeArray(
		Graphics::D3D12::Device* device,
		const TextureCubeDescription& desc,
		const TextureResource* init_data,
		ID3D12GraphicsCommandList* cmd_queue)
	: Texture(device, desc, init_data, cmd_queue)
	{
	}

	TextureCubeArray::TextureCubeArray(const TextureCubeArray& rhs)
	: Texture(rhs)
	{
	}

	std::unique_ptr<Runtime::Texture> TextureCubeArray::clone() const
	{
		return std::make_unique<TextureCubeArray>(*this);
	}
}}}}
