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
#include <vcl/config/direct3d12.h>

// C++ standard library
#include <chrono>

// Windows Runtime Library
#include <wrl.h>

// VCL
#include <vcl/graphics/d3d12/semaphore.h>

namespace Vcl { namespace Graphics { namespace D3D12
{
	class Device;

	class CommandQueue
	{
	public:
		template<typename T>
		using ComPtr = Microsoft::WRL::ComPtr<T>;

		CommandQueue(Device* device);

		//! Native queue pointer
		ID3D12CommandQueue* nativeQueue() const { return _d3d12Queue.Get(); }

		//! Wait until the queue has been fully processed
		//! \param duration Time to wait in milliseconds
		void sync(std::chrono::milliseconds duration = std::chrono::milliseconds::max());

	private:
		//! Associated device
		Device* _device;

		//! Native queue object
		ComPtr<ID3D12CommandQueue> _d3d12Queue;

		//! Semaphore for CPU-GPU synchronization 
		Semaphore _semaphore;
	};
}}}
