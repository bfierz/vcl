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
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>

// C++ Standard Library
#include <utility>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/3rdparty/d3dx12.h>
#include <vcl/graphics/d3d12/d3d.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 {
	void Resource::transition(ID3D12GraphicsCommandList* cmd_list, D3D12_RESOURCE_STATES target_state)
	{
		if (_currentStates != target_state)
		{
			CD3DX12_RESOURCE_BARRIER barriers[] = {
				CD3DX12_RESOURCE_BARRIER::Transition(
					handle(),
					_currentStates, target_state)
			};
			cmd_list->ResourceBarrier(1, barriers);
			_currentStates = target_state;
		}
	}

	D3D12_RESOURCE_STATES toD3DResourceState(Flags<BufferUsage> flag)
	{
		// clang-format off
		UINT d3d_flags = 0;
		d3d_flags |= (flag.isSet(BufferUsage::CopySrc))   ? D3D12_RESOURCE_STATE_COPY_SOURCE : 0;
		d3d_flags |= (flag.isSet(BufferUsage::CopyDst))   ? D3D12_RESOURCE_STATE_COPY_DEST : 0;
		d3d_flags |= (flag.isSet(BufferUsage::Index))     ? D3D12_RESOURCE_STATE_INDEX_BUFFER : 0;
		d3d_flags |= (flag.isSet(BufferUsage::Vertex))    ? D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER : 0;
		d3d_flags |= (flag.isSet(BufferUsage::Uniform))   ? D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER : 0;
		d3d_flags |= (flag.isSet(BufferUsage::Storage))   ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : 0;
		d3d_flags |= (flag.isSet(BufferUsage::Indirect))  ? D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT : 0;
		d3d_flags |= (flag.isSet(BufferUsage::StreamOut)) ? D3D12_RESOURCE_STATE_STREAM_OUT : 0;
		// clang-format on

		return (D3D12_RESOURCE_STATES)d3d_flags;
	}

	//! Determine the heap-type according to the buffer usage.
	//! Mappable buffers are mutually exclusive and have dedicated heap-types
	D3D12_HEAP_TYPE determineHeapType(Flags<BufferUsage> usage)
	{
		if (usage.isSet(BufferUsage::MapRead))
			return D3D12_HEAP_TYPE_READBACK;
		else if (usage.isSet(BufferUsage::MapWrite))
			return D3D12_HEAP_TYPE_UPLOAD;
		else
			return D3D12_HEAP_TYPE_DEFAULT;
	}

	Buffer::Buffer(Graphics::D3D12::Device* device, const BufferDescription& desc, const BufferInitData* init_data, ID3D12GraphicsCommandList* cmd_queue)
	: Runtime::Buffer(desc.SizeInBytes, desc.Usage)
	{
		VclRequire(implies(usage().isSet(BufferUsage::MapRead), !usage().isSet(BufferUsage::MapWrite)), "Buffer mappable for reading is not mappable for writing");
		VclRequire(implies(usage().isSet(BufferUsage::MapWrite), !usage().isSet(BufferUsage::MapRead)), "Buffer mappable for writing is not mappable for reading");
		VclRequire(implies(usage().isSet(BufferUsage::MapRead) || usage().isSet(BufferUsage::MapWrite), init_data == nullptr), "Mappable buffer does not have initial data");

		// D3D12_HEAP_TYPE_READBACK and D3D12_HEAP_TYPE_UPLOAD require specific usage flags
		// https://docs.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_type
		_heapType = determineHeapType(usage());
		D3D12_RESOURCE_STATES buffer_usage;
		if (_heapType == D3D12_HEAP_TYPE_READBACK)
		{
			buffer_usage = D3D12_RESOURCE_STATE_COPY_DEST;
			_targetStates = buffer_usage;
		} else if (_heapType == D3D12_HEAP_TYPE_UPLOAD)

		{
			buffer_usage = D3D12_RESOURCE_STATE_GENERIC_READ;
			_targetStates = buffer_usage;
		} else
		{
			buffer_usage = D3D12_RESOURCE_STATE_COMMON;
			_targetStates = toD3DResourceState(usage());
		}

		_currentStates = init_data ? D3D12_RESOURCE_STATE_COPY_DEST : buffer_usage;

		ID3D12Device* d3d12_dev = device->nativeDevice();
		D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
		if (usage().isSet(BufferUsage::Storage))
			flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		const auto heap_props = CD3DX12_HEAP_PROPERTIES(_heapType);
		const auto buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(desc.SizeInBytes, flags);
		VCL_DIRECT3D_SAFE_CALL(d3d12_dev->CreateCommittedResource(
			&heap_props,
			D3D12_HEAP_FLAG_NONE,
			&buffer_desc,
			_currentStates,
			nullptr,
			IID_PPV_ARGS(&_resource)));

		if (init_data)
		{
			const auto upload_heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
			const auto upload_buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(desc.SizeInBytes);
			VCL_DIRECT3D_SAFE_CALL(d3d12_dev->CreateCommittedResource(
				&upload_heap_props,
				D3D12_HEAP_FLAG_NONE,
				&upload_buffer_desc,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&_uploadResource)));

			D3D12_SUBRESOURCE_DATA subresource_data = {};
			subresource_data.pData = init_data->Data;
			subresource_data.RowPitch = desc.SizeInBytes;
			subresource_data.SlicePitch = subresource_data.RowPitch;

			UpdateSubresources(cmd_queue, _resource.Get(), _uploadResource.Get(), 0, 0, 1, &subresource_data);
		}
	}

	Buffer::~Buffer()
	{
	}

	void Buffer::unmap(D3D12_RANGE range) const
	{
		_resource->Unmap(0, range.Begin == range.End ? nullptr : &range);
	}

	void* Buffer::map(D3D12_RANGE range) const
	{
		VclRequire(usage().isSet(BufferUsage::MapRead) | usage().isSet(BufferUsage::MapWrite), "Buffer is mappable.");
		void* ptr = nullptr;
		VCL_DIRECT3D_SAFE_CALL(_resource->Map(0, range.Begin == range.End ? nullptr : &range, &ptr));

		return ptr;
	}

	void Buffer::write(Graphics::D3D12::Device* device, ID3D12GraphicsCommandList* cmd_list, const void* data, size_t offset_in_bytes, size_t size_in_bytes)
	{
		VclRequire(offset_in_bytes + size_in_bytes <= sizeInBytes(), "Data fits into buffer.");

		if (usage().isSet(BufferUsage::MapWrite))
		{
			auto ptr = reinterpret_cast<uint8_t*>(map());
			memcpy(ptr + offset_in_bytes, data, size_in_bytes);
			unmap({ offset_in_bytes, size_in_bytes });
		} else
		{
			if (!_uploadResource)
			{
				const auto upload_heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
				const auto upload_buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes());
				VCL_DIRECT3D_SAFE_CALL(device->nativeDevice()->CreateCommittedResource(
					&upload_heap_props,
					D3D12_HEAP_FLAG_NONE,
					&upload_buffer_desc,
					D3D12_RESOURCE_STATE_GENERIC_READ,
					nullptr,
					IID_PPV_ARGS(&_uploadResource)));
			}

			uint8_t* ptr;
			VCL_DIRECT3D_SAFE_CALL(_uploadResource->Map(0, nullptr, reinterpret_cast<void**>(&ptr)));

			memcpy(ptr + offset_in_bytes, data, size_in_bytes);
			D3D12_RANGE written_bytes{ offset_in_bytes, offset_in_bytes + size_in_bytes };
			_uploadResource->Unmap(0, &written_bytes);

			D3D12_SUBRESOURCE_DATA subresource_data = {};
			subresource_data.pData = data;
			subresource_data.RowPitch = size_in_bytes;
			subresource_data.SlicePitch = subresource_data.RowPitch;

			cmd_list->CopyBufferRegion(
				_resource.Get(), offset_in_bytes, _uploadResource.Get(), 0, size_in_bytes);
		}
	}
	/*
	void Buffer::copyTo(ID3D12DeviceContext* ctx, void* dst, size_t srcOffset, size_t dstOffset, size_t size) const
	{
		VclRequire(_buffer != nullptr, "D3D12 buffer is created.");
		VclRequire(implies(size < std::numeric_limits<size_t>::max(), srcOffset + size <= sizeInBytes()), "Size to copy is valid");

		if (size == std::numeric_limits<size_t>::max())
			size = sizeInBytes() - srcOffset;

		const auto src = map(ctx, MapOptions::Read);
		memcpy(static_cast<char*>(dst) + dstOffset, static_cast<char*>(src) + srcOffset, size);
		unmap(ctx);
	}

	void Buffer::copyTo(ID3D12DeviceContext* ctx, Buffer& target, size_t srcOffset, size_t dstOffset, size_t size) const
	{
		VclRequire(_buffer != nullptr, "D3D12 buffer is created.");
		VclRequire(target.id() > 0, "GL buffer is created.");
		VclRequire(sizeInBytes() <= target.sizeInBytes(), "Size to copy is valid");
		VclRequire(implies(size < std::numeric_limits<size_t>::max(), srcOffset + size <= sizeInBytes()), "Size to copy is valid");
		VclRequire(implies(size < std::numeric_limits<size_t>::max(), dstOffset + size <= target.sizeInBytes()), "Size to copy is valid");

		if (size == std::numeric_limits<size_t>::max())
			size = sizeInBytes() - srcOffset;
		VclCheck(dstOffset + size <= target.sizeInBytes(), "Size to copy is valid");

		D3D12_BOX src_region;
		ZeroMemory(&src_region, sizeof(D3D12_BOX));
		src_region.left = srcOffset;
		src_region.right = srcOffset + size;
		ctx->CopySubresourceRegion(target.handle(), 0, dstOffset, 0, 0, _buffer, 0, &src_region);
	}*/
}}}}
