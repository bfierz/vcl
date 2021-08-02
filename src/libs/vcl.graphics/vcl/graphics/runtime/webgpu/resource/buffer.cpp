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
#include <vcl/graphics/runtime/webgpu/resource/buffer.h>

// C++ Standard Library
#include <utility>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU {
	WGPUBufferUsageFlags toWGPU(Flags<BufferUsage> flags)
	{
		WGPUBufferUsageFlags wgpu_flags = 0;
		wgpu_flags |= (flags.isSet(BufferUsage::MapRead))  ? WGPUBufferUsage_MapRead  : 0;
		wgpu_flags |= (flags.isSet(BufferUsage::MapWrite)) ? WGPUBufferUsage_MapWrite : 0;
		wgpu_flags |= (flags.isSet(BufferUsage::CopySrc))  ? WGPUBufferUsage_CopySrc  : 0;
		wgpu_flags |= (flags.isSet(BufferUsage::CopyDst))  ? WGPUBufferUsage_CopyDst  : 0;
		wgpu_flags |= (flags.isSet(BufferUsage::Index))    ? WGPUBufferUsage_Index    : 0;
		wgpu_flags |= (flags.isSet(BufferUsage::Vertex))   ? WGPUBufferUsage_Vertex   : 0;
		wgpu_flags |= (flags.isSet(BufferUsage::Uniform))  ? WGPUBufferUsage_Uniform  : 0;
		wgpu_flags |= (flags.isSet(BufferUsage::Storage))  ? WGPUBufferUsage_Storage  : 0;
		wgpu_flags |= (flags.isSet(BufferUsage::Indirect)) ? WGPUBufferUsage_Indirect : 0;

		VclCheck(!flags.isSet(BufferUsage::StreamOut), "Stream-out is not supported");

		return wgpu_flags;
	}

	Buffer::Buffer(WGPUDevice device, const BufferDescription& desc, const BufferInitData* init_data, WGPUQueue queue_id)
	: Runtime::Buffer(desc.SizeInBytes, desc.Usage)
	{
		VclRequire(implies(usage().isSet(BufferUsage::MapRead), !usage().isSet(BufferUsage::MapWrite)), "Buffer mappable for reading is not mappable for writing");
		VclRequire(implies(usage().isSet(BufferUsage::MapWrite), !usage().isSet(BufferUsage::MapRead)), "Buffer mappable for writing is not mappable for reading");
		VclRequire(implies(usage().isSet(BufferUsage::MapRead) || usage().isSet(BufferUsage::MapWrite), init_data == nullptr), "Mappable buffer does not have initial data");

		WGPUBufferDescriptor wgpu_desc = {
			nullptr,
			"Generic buffer",
			toWGPU(desc.Usage),
			desc.SizeInBytes,
			init_data != nullptr
		} ;

		_resource = wgpuDeviceCreateBuffer(device, &wgpu_desc);

		if (init_data)
		{
			wgpuQueueWriteBuffer(queue_id,
				_resource,
				0,
				reinterpret_cast<const uint8_t*>(init_data->Data),
				init_data->SizeInBytes);
		}
	}

	Buffer::~Buffer()
	{
	}

	void Buffer::unmap() const
	{
		wgpuBufferUnmap(_resource);
	}
	
	void* Buffer::map() const
	{
		VclRequire(usage().isSet(BufferUsage::MapRead) | usage().isSet(BufferUsage::MapWrite), "Buffer is mappable.");
		WGPUMapMode mode = usage().isSet(BufferUsage::MapRead) ? WGPUMapMode_Read : WGPUMapMode_Write;
		wgpuBufferMapAsync(_resource, mode, 0, sizeInBytes(), nullptr, nullptr);
		if (mode == WGPUMapMode_Read)
			return const_cast<void*>(wgpuBufferGetConstMappedRange(_resource, 0, sizeInBytes()));
		else
			return wgpuBufferGetMappedRange(_resource, 0, sizeInBytes());
	}

	void Buffer::copyTo(WGPUCommandEncoder cmd_encoder, Buffer tgt)
	{
		wgpuCommandEncoderCopyBufferToBuffer(cmd_encoder, _resource, 0, tgt.handle(), 0, sizeInBytes());
	}

	//void Buffer::write(Graphics::D3D12::Device* device, ID3D12GraphicsCommandList* cmd_list, const void* data, size_t offset_in_bytes, size_t size_in_bytes)
	//{
	//	VclRequire(offset_in_bytes + size_in_bytes <= sizeInBytes(), "Data fits into buffer.");
	//
	//	if (usage().isSet(BufferUsage::MapWrite))
	//	{
	//		auto ptr = reinterpret_cast<uint8_t*>(map());
	//		memcpy(ptr + offset_in_bytes, data, size_in_bytes);
	//		unmap({ offset_in_bytes, size_in_bytes });
	//	}
	//	else
	//	{
	//		if (!_uploadResource)
	//		{
	//			VCL_DIRECT3D_SAFE_CALL(device->nativeDevice()->CreateCommittedResource(
	//				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
	//				D3D12_HEAP_FLAG_NONE,
	//				&CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes()),
	//				D3D12_RESOURCE_STATE_GENERIC_READ,
	//				nullptr,
	//				IID_PPV_ARGS(&_uploadResource)));
	//		}
	//
	//		uint8_t* ptr;
	//		VCL_DIRECT3D_SAFE_CALL(_uploadResource->Map(0, nullptr, reinterpret_cast<void**>(&ptr)));
	//
	//		memcpy(ptr + offset_in_bytes, data, size_in_bytes);
	//		D3D12_RANGE written_bytes{ offset_in_bytes, offset_in_bytes + size_in_bytes };
	//		_uploadResource->Unmap(0, &written_bytes);
	//
	//		D3D12_SUBRESOURCE_DATA subresource_data = {};
	//		subresource_data.pData = data;
	//		subresource_data.RowPitch = size_in_bytes;
	//		subresource_data.SlicePitch = subresource_data.RowPitch;
	//
	//		cmd_list->CopyBufferRegion(
	//			_resource.Get(), offset_in_bytes, _uploadResource.Get(), 0, size_in_bytes);
	//	}
	//}
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
