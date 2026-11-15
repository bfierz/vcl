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
#include <wrl/client.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	//! Fence with a host event to wait for it
	class Semaphore
	{
	public:
		template<typename T>
		using ComPtr = Microsoft::WRL::ComPtr<T>;

		Semaphore(ID3D12Device* device);
		Semaphore(const Semaphore&) = delete;
		Semaphore& operator=(const Semaphore&) = delete;
		~Semaphore();

		ID3D12Fence* nativeFence() const { return _d3d12Fence.Get(); }

		//! Signals the next value from 'queue'
		//! \returns the signaled value
		uint64_t signal(ID3D12CommandQueue* queue);

		//! Waits until the fence reaches 'value'
		//! \returns false if 'duration' passed before the fence reached 'value'
		bool wait(uint64_t value, std::chrono::milliseconds duration = std::chrono::milliseconds::max());
#ifdef _WIN32
		//! Waits until the fence reaches 'value' and 'additional_event' is signaled
		void wait(uint64_t value, HANDLE additional_event, std::chrono::milliseconds duration = std::chrono::milliseconds::max());
#endif

		//! Signals the next value from 'queue' and waits for it
		//! \returns false if 'duration' passed before the queue reached the signal
		bool sync(ID3D12CommandQueue* queue, std::chrono::milliseconds duration = std::chrono::milliseconds::max());

	private:
		//! Waits until '_event' is signaled
		//! \returns false if 'duration' passed first
		bool waitForEvent(std::chrono::milliseconds duration);

		ComPtr<ID3D12Fence> _d3d12Fence;

		//! Host event signaled by the fence. An eventfd on Linux, as WSL's D3D12 expects.
		HANDLE _event{ nullptr };
#ifndef _WIN32
		int _eventFd{ -1 };
#endif
		uint64_t _value{ 0 };
	};
}}}
