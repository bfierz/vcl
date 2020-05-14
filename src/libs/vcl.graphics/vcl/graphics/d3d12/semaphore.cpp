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
#include <vcl/graphics/d3d12/semaphore.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/d3d.h>

namespace Vcl { namespace Graphics { namespace D3D12
{
	Semaphore::Semaphore(ID3D12Device* device)
	{
		VCL_DIRECT3D_SAFE_CALL(device->CreateFence(_value, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&_d3d12Fence)));
		_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);

		VclEnsure(_d3d12Fence, "GPU event created");
		VclEnsure(_event, "Host event created");
	}

	uint64_t Semaphore::signal(ID3D12CommandQueue* queue)
	{
		uint64_t new_value = ++_value;
		VCL_DIRECT3D_SAFE_CALL(queue->Signal(_d3d12Fence.Get(), new_value));

		return new_value;
	}

	void Semaphore::wait(uint64_t value, std::chrono::milliseconds duration)
	{
		if (_d3d12Fence->GetCompletedValue() < value)
		{
			VCL_DIRECT3D_SAFE_CALL(_d3d12Fence->SetEventOnCompletion(value, _event));
			::WaitForSingleObject(_event, static_cast<DWORD>(duration.count()));
		}
	}

	void Semaphore::wait(uint64_t value, HANDLE additional_event, std::chrono::milliseconds duration)
	{
		if (_d3d12Fence->GetCompletedValue() < value)
		{
			VCL_DIRECT3D_SAFE_CALL(_d3d12Fence->SetEventOnCompletion(value, _event));

			HANDLE waitableObjects[] = { additional_event, _event };
			::WaitForMultipleObjects(2, waitableObjects, TRUE, static_cast<DWORD>(duration.count()));
		}
		else
		{
			::WaitForSingleObject(additional_event, static_cast<DWORD>(duration.count()));
		}
	}

	void Semaphore::sync(ID3D12CommandQueue* queue, std::chrono::milliseconds duration)
	{
		const auto sig_value = signal(queue);
		wait(sig_value);
	}
}}}
