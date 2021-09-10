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
#include <vcl/config/eigen.h>
#include <vcl/config/webgpu.h>

// VCL
#include <vcl/core/flags.h>
#include <vcl/graphics/runtime/resource/buffer.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU {
	class Buffer : public Runtime::Buffer
	{
	public:
		Buffer(WGPUDevice device, const BufferDescription& desc, const BufferInitData* init_data = nullptr, WGPUQueue queue_id = 0);
		virtual ~Buffer();

		//! Resource handle
		WGPUBuffer handle() const { return _resource; }

		//! Map the device memory to a host visible address
		void* map() const;

		//! Unmap the buffer
		void unmap() const;

		//void write(Graphics::D3D12::Device* device, ID3D12GraphicsCommandList* cmd_list, const void* data, size_t offset_in_bytes, size_t size_in_bytes);

		//! \defgroup Data copy methods
		//! \{

		void copyTo(WGPUCommandEncoder cmd_encoder, Buffer tgt);
		//void copyTo(ID3D11DeviceContext* ctx, void* dst, size_t srcOffset = 0, size_t dstOffset = 0, size_t size = std::numeric_limits<size_t>::max()) const;
		//void copyTo(ID3D11DeviceContext* ctx, Buffer& target, size_t srcOffset = 0, size_t dstOffset = 0, size_t size = std::numeric_limits<size_t>::max()) const;

		//! \}

	private:
		//!
		WGPUBuffer _resource;
	};
}}}}
