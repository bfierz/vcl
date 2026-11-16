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
#include <vcl/graphics/d3d12/descriptorheap.h>

// C++ standard library
#include <algorithm>

// VCL
#include <vcl/core/contract.h>
#include <directx/d3dx12.h>
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/runtime/d3d12/state/sampler.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	using Runtime::ResourceIndex;
	using Runtime::ResourceViewType;
	using Runtime::SamplerIndex;
	using Runtime::TextureType;
	using Runtime::TextureViewDescription;

	namespace {
		bool isBufferView(ResourceViewType type)
		{
			return type == ResourceViewType::ConstantBuffer || type == ResourceViewType::Buffer || type == ResourceViewType::RWBuffer;
		}

		bool isArrayType(TextureType type)
		{
			return type == TextureType::Texture1DArray || type == TextureType::Texture2DArray || type == TextureType::Texture2DMSArray || type == TextureType::TextureCubeArray;
		}

		//! Part of a texture a view covers, with the defaults resolved
		struct TextureRange
		{
			TextureType Type;
			DXGI_FORMAT Format;
			uint32_t FirstMipLevel;
			uint32_t MipLevels;
			uint32_t FirstLayer;
			uint32_t Layers;
		};

		D3D12_SHADER_RESOURCE_VIEW_DESC toShaderResourceView(const TextureRange& range)
		{
			D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
			desc.Format = toShaderResourceFormat(range.Format);
			desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			switch (range.Type)
			{
			case TextureType::Texture1D:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
				desc.Texture1D = { range.FirstMipLevel, range.MipLevels, 0.0f };
				break;
			case TextureType::Texture1DArray:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
				desc.Texture1DArray = { range.FirstMipLevel, range.MipLevels, range.FirstLayer, range.Layers, 0.0f };
				break;
			case TextureType::Texture2D:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
				desc.Texture2D = { range.FirstMipLevel, range.MipLevels, 0, 0.0f };
				break;
			case TextureType::Texture2DArray:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
				desc.Texture2DArray = { range.FirstMipLevel, range.MipLevels, range.FirstLayer, range.Layers, 0, 0.0f };
				break;
			case TextureType::Texture2DMS:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
				break;
			case TextureType::Texture2DMSArray:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
				desc.Texture2DMSArray = { range.FirstLayer, range.Layers };
				break;
			case TextureType::Texture3D:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
				desc.Texture3D = { range.FirstMipLevel, range.MipLevels, 0.0f };
				break;
			case TextureType::TextureCube:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
				desc.TextureCube = { range.FirstMipLevel, range.MipLevels, 0.0f };
				break;
			case TextureType::TextureCubeArray:
				desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
				desc.TextureCubeArray = { range.FirstMipLevel, range.MipLevels, range.FirstLayer, range.Layers / 6, 0.0f };
				break;
			case TextureType::Unknown:
				VclDebugError("Texture type is known.");
				break;
			}

			return desc;
		}

		D3D12_UNORDERED_ACCESS_VIEW_DESC toUnorderedAccessView(const TextureRange& range)
		{
			D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
			desc.Format = range.Format;
			switch (range.Type)
			{
			case TextureType::Texture1D:
				desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
				desc.Texture1D = { range.FirstMipLevel };
				break;
			case TextureType::Texture1DArray:
				desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
				desc.Texture1DArray = { range.FirstMipLevel, range.FirstLayer, range.Layers };
				break;
			case TextureType::Texture2D:
				desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
				desc.Texture2D = { range.FirstMipLevel, 0 };
				break;
			case TextureType::Texture2DArray:
			case TextureType::TextureCube:
			case TextureType::TextureCubeArray:
				// Cube maps are written as arrays of 2D textures
				desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
				desc.Texture2DArray = { range.FirstMipLevel, range.FirstLayer, range.Layers, 0 };
				break;
			case TextureType::Texture3D:
				desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
				desc.Texture3D = { range.FirstMipLevel, 0, ~0u };
				break;
			case TextureType::Texture2DMS:
			case TextureType::Texture2DMSArray:
			case TextureType::Unknown:
				VclDebugError("Texture type supports read-write views.");
				break;
			}

			return desc;
		}
	}

	DescriptorHeap::DescriptorHeap(Device* device, const Runtime::BindlessLayoutDescription& desc)
	: _device(device)
	, _desc(desc)
	, _resources(desc.ResourceCapacity)
	, _samplers(desc.SamplerCapacity)
	, _viewTypes(desc.ResourceCapacity, ResourceViewType::ConstantBuffer)
	{
		VclRequire(device, "Device is set.");
		VclRequire(Runtime::validate(desc, device->features()).empty(), "Layout is supported by the device.");

		auto d3d12_device = device->nativeDevice();

		// Root signature shared by all pipelines
		{
			CD3DX12_ROOT_PARAMETER1 root_constants;
			root_constants.InitAsConstants(desc.PushConstantSize / 4, 0, 0, D3D12_SHADER_VISIBILITY_ALL);

			std::vector<D3D12_STATIC_SAMPLER_DESC> static_samplers;
			static_samplers.reserve(desc.StaticSamplers.size());
			std::transform(desc.StaticSamplers.begin(), desc.StaticSamplers.end(), std::back_inserter(static_samplers), [](const Runtime::StaticSampler& sampler) {
				return Runtime::D3D12::toD3D12(sampler);
			});

			const D3D12_ROOT_SIGNATURE_FLAGS flags =
				D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
				D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
				D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;

			CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC sig_desc;
			sig_desc.Init_1_1(
				desc.PushConstantSize > 0 ? 1 : 0, &root_constants,
				static_cast<UINT>(static_samplers.size()), static_samplers.data(),
				flags);
			_rootSignature = device->createRootSignature(sig_desc);
			VclCheck(_rootSignature, "Root signature was created.");
		}

		// Shader-visible heaps
		_resourceHeap = device->createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, desc.ResourceCapacity, true);
		_samplerHeap = device->createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, desc.SamplerCapacity, true);
		VclCheck(_resourceHeap && _samplerHeap, "Heaps were created.");

		_resourceHeapStart = cpuHeapStart(_resourceHeap.Get());
		_samplerHeapStart = cpuHeapStart(_samplerHeap.Get());
		_resourceIncrement = d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		_samplerIncrement = d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
	}

	DescriptorHeap::~DescriptorHeap()
	{
		VclRequire(_resources.nrAllocated() == 0, "All resource views are released.");
		VclRequire(_samplers.nrAllocated() == 0, "All samplers are released.");
	}

	ResourceIndex DescriptorHeap::allocateResource(ResourceViewType type)
	{
		const ResourceIndex index{ _resources.allocate() };
		VclCheck(index.isValid(), "Resource heap has a free slot.");
		if (index.isValid())
			_viewTypes[index.Value] = type;

		return index;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE DescriptorHeap::resourceHandle(ResourceIndex index) const
	{
		return CD3DX12_CPU_DESCRIPTOR_HANDLE(_resourceHeapStart, static_cast<INT>(index.Value), _resourceIncrement);
	}

	ResourceView DescriptorHeap::createView(const Runtime::D3D12::Buffer& buffer, ResourceViewType type, uint64_t offset, uint64_t size, uint32_t stride)
	{
		using Runtime::BufferUsage;

		// Constant buffer views cover multiples of 256 bytes
		const uint64_t cbv_alignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
		const uint64_t cbv_size = (size + cbv_alignment - 1) / cbv_alignment * cbv_alignment;
		const uint64_t resource_size = resourceDesc(buffer.handle()).Width;

		VclRequire(isBufferView(type), "View type is a buffer type.");
		VclRequire(size > 0 && offset + size <= buffer.sizeInBytes(), "Range is inside the buffer.");
		VclRequire(implies(type == ResourceViewType::ConstantBuffer, buffer.usage().isSet(BufferUsage::Uniform)), "Buffer is usable as constant buffer.");
		VclRequire(implies(type != ResourceViewType::ConstantBuffer, buffer.usage().isSet(BufferUsage::Storage)), "Buffer is usable as storage buffer.");
		VclRequire(implies(type == ResourceViewType::ConstantBuffer, offset % cbv_alignment == 0), "Constant buffers are aligned to 256 bytes.");
		VclRequire(implies(type == ResourceViewType::ConstantBuffer, offset + cbv_size <= resource_size), "Aligned constant buffer range is inside the buffer.");
		VclRequire(implies(stride > 0, offset % stride == 0 && size % stride == 0), "Range covers entire elements.");
		VclRequire(implies(stride == 0 && type != ResourceViewType::ConstantBuffer, offset % 4 == 0 && size % 4 == 0), "Raw range covers entire 32-bit values.");
		VCL_UNREFERENCED_PARAMETER(resource_size);

		const ResourceIndex index = allocateResource(type);
		if (!index.isValid())
			return {};

		auto d3d12_device = _device->nativeDevice();
		const auto handle = resourceHandle(index);
		const uint64_t element_size = stride > 0 ? stride : 4;
		if (type == ResourceViewType::ConstantBuffer)
		{
			D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};
			desc.BufferLocation = buffer.handle()->GetGPUVirtualAddress() + offset;
			desc.SizeInBytes = static_cast<UINT>(cbv_size);
			d3d12_device->CreateConstantBufferView(&desc, handle);
		} else if (type == ResourceViewType::Buffer)
		{
			D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
			desc.Format = stride > 0 ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_TYPELESS;
			desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
			desc.Buffer.FirstElement = offset / element_size;
			desc.Buffer.NumElements = static_cast<UINT>(size / element_size);
			desc.Buffer.StructureByteStride = stride;
			desc.Buffer.Flags = stride > 0 ? D3D12_BUFFER_SRV_FLAG_NONE : D3D12_BUFFER_SRV_FLAG_RAW;
			d3d12_device->CreateShaderResourceView(buffer.handle(), &desc, handle);
		} else
		{
			D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
			desc.Format = stride > 0 ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_TYPELESS;
			desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			desc.Buffer.FirstElement = offset / element_size;
			desc.Buffer.NumElements = static_cast<UINT>(size / element_size);
			desc.Buffer.StructureByteStride = stride;
			desc.Buffer.CounterOffsetInBytes = 0;
			desc.Buffer.Flags = stride > 0 ? D3D12_BUFFER_UAV_FLAG_NONE : D3D12_BUFFER_UAV_FLAG_RAW;
			d3d12_device->CreateUnorderedAccessView(buffer.handle(), nullptr, &desc, handle);
		}

		return { this, index };
	}

	ResourceView DescriptorHeap::createView(const Runtime::D3D12::Texture& texture, ResourceViewType type, const TextureViewDescription& view)
	{
		using Runtime::TextureUsage;

		const bool read_write = type == ResourceViewType::RWTexture;
		const auto view_type = view.Type != TextureType::Unknown ? view.Type : texture.type();
		const uint32_t nr_mips = view.MipLevels != TextureViewDescription::Remaining ? view.MipLevels : static_cast<uint32_t>(texture.mipMapLevels()) - view.FirstMipLevel;
		const uint32_t nr_layers = view.Layers != TextureViewDescription::Remaining ? view.Layers : static_cast<uint32_t>(texture.layers()) - view.FirstLayer;
		const bool is_cube = view_type == TextureType::TextureCube || view_type == TextureType::TextureCubeArray;

		VclRequire(!isBufferView(type), "View type is a texture type.");
		VclRequire(implies(!read_write, texture.usage().isSet(TextureUsage::Sampled)), "Texture is usable as sampled texture.");
		VclRequire(implies(read_write, texture.usage().isSet(TextureUsage::Storage)), "Texture is usable as storage texture.");
		VclRequire(nr_mips > 0 && view.FirstMipLevel + nr_mips <= static_cast<uint32_t>(texture.mipMapLevels()), "Mip-map levels are inside the texture.");
		VclRequire(nr_layers > 0 && view.FirstLayer + nr_layers <= static_cast<uint32_t>(texture.layers()), "Layers are inside the texture.");
		VclRequire(implies(read_write, nr_mips == 1), "Read-write views cover one mip-map level.");
		VclRequire(texture.supportsViewFormat(view.Format), "View format is the format of the texture or one of its view formats.");
		VclRequire(implies(!isArrayType(view_type) && !is_cube && view_type != TextureType::Texture3D, view.FirstLayer == 0 && nr_layers == 1), "Non-array views cover the first layer (D3D12 can't offset them).");
		VclRequire(implies(view_type == TextureType::TextureCube, view.FirstLayer == 0 && nr_layers == 6), "Cube views cover the first six layers.");
		VclRequire(implies(view_type == TextureType::TextureCubeArray, nr_layers % 6 == 0), "Cube-array views cover entire cubes.");

		const ResourceIndex index = allocateResource(type);
		if (!index.isValid())
			return {};

		const TextureRange range = {
			view_type,
			D3D::toD3Denum(view.Format != SurfaceFormat::Unknown ? view.Format : texture.format()),
			view.FirstMipLevel,
			nr_mips,
			view.FirstLayer,
			nr_layers
		};

		auto d3d12_device = _device->nativeDevice();
		if (read_write)
		{
			const auto desc = toUnorderedAccessView(range);
			d3d12_device->CreateUnorderedAccessView(texture.handle(), nullptr, &desc, resourceHandle(index));
		} else
		{
			const auto desc = toShaderResourceView(range);
			d3d12_device->CreateShaderResourceView(texture.handle(), &desc, resourceHandle(index));
		}

		return { this, index };
	}

	SamplerView DescriptorHeap::createSampler(const Runtime::SamplerDescription& desc)
	{
		const SamplerIndex index{ _samplers.allocate() };
		VclCheck(index.isValid(), "Sampler heap has a free slot.");
		if (!index.isValid())
			return {};

		const D3D12_SAMPLER_DESC sampler_desc = Runtime::D3D12::toD3D12(desc);
		const CD3DX12_CPU_DESCRIPTOR_HANDLE handle(_samplerHeapStart, static_cast<INT>(index.Value), _samplerIncrement);
		_device->nativeDevice()->CreateSampler(&sampler_desc, handle);

		return { this, index };
	}

	Runtime::ResourceViewType DescriptorHeap::viewType(ResourceIndex index) const
	{
		VclRequire(index.Value < _viewTypes.size(), "Index is valid.");
		return _viewTypes[index.Value];
	}

	void DescriptorHeap::retire(ResourceIndex index)
	{
		_resources.retire(index.Value);
	}

	void DescriptorHeap::retire(SamplerIndex index)
	{
		_samplers.retire(index.Value);
	}

	void DescriptorHeap::advance(uint64_t next_fence_value, uint64_t completed_fence_value)
	{
		// Descriptors own no objects. The stale descriptors stay in the heap,
		// which is valid as long as no shader accesses them.
		_resources.advance(next_fence_value, completed_fence_value);
		_samplers.advance(next_fence_value, completed_fence_value);
	}

	void DescriptorHeap::bind(ID3D12GraphicsCommandList* cmd_list) const
	{
		VclRequire(cmd_list, "Command list is set.");

		// Directly indexed heaps must be set before the root signature
		ID3D12DescriptorHeap* heaps[] = { _resourceHeap.Get(), _samplerHeap.Get() };
		cmd_list->SetDescriptorHeaps(2, heaps);

		// Compute command lists don't support graphics state
		if (cmd_list->GetType() != D3D12_COMMAND_LIST_TYPE_COMPUTE)
			cmd_list->SetGraphicsRootSignature(_rootSignature.Get());
		cmd_list->SetComputeRootSignature(_rootSignature.Get());
	}
}}}
