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
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/resource/buffer.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU {
	class Buffer : public Runtime::Buffer
	{
	public:
		Buffer(WGPUDevice device, const BufferDescription& desc);
		Buffer(WGPUDevice device, const BufferDescription& desc, const BufferInitData* init_data, WGPUQueue queue_id);
		template<typename T>
		Buffer(WGPUDevice device, const BufferDescription& desc, stdext::span<const T> init_data, WGPUQueue queue_id)
		: Buffer(device, desc)
		{
			BufferInitData init_data_desc = {
				init_data.data(),
				init_data.size() * sizeof(T)
			};
			initializeData(&init_data_desc, queue_id);
		}
		template<typename T>
		Buffer(WGPUDevice device, const BufferDescription& desc, stdext::span<T> init_data, WGPUQueue queue_id)
		: Buffer(device, desc)
		{
			BufferInitData init_data_desc = {
				init_data.data(),
				init_data.size() * sizeof(T)
			};
			initializeData(&init_data_desc, queue_id);
		}
		~Buffer() override;

		//! Resource handle
		WGPUBuffer handle() const { return _resource; }

		//! Map the device memory to a host visible address
		void* map() const;

		//! Unmap the buffer
		void unmap() const;

		//! Copy the content of this buffer to the target \p tgt
		void copyTo(WGPUCommandEncoder cmd_encoder, Buffer tgt);

	private:
		void initializeData(const BufferInitData* init_data, WGPUQueue queue_id);

		//! WebGPU resource
		WGPUBuffer _resource;
	};
}}}}
