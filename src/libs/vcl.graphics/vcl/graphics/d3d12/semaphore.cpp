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

// C++ standard library
#include <algorithm>
#include <climits>

// C runtime
#ifndef _WIN32
#	include <cerrno>
#	include <fcntl.h>
#	include <poll.h>
#	include <sys/eventfd.h>
#	include <unistd.h>
#endif

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/d3d.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	namespace {
		constexpr auto Infinite = std::chrono::milliseconds::max();

#ifdef _WIN32
		DWORD toTimeout(std::chrono::milliseconds duration)
		{
			return duration == Infinite ? INFINITE : static_cast<DWORD>(std::min<long long>(duration.count(), INFINITE - 1));
		}
#endif
	}

	Semaphore::Semaphore(ID3D12Device* device)
	{
		VCL_DIRECT3D_SAFE_CALL(device->CreateFence(_value, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&_d3d12Fence)));
#ifdef _WIN32
		_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);
#else
		_eventFd = eventfd(0, EFD_CLOEXEC);
		if (_eventFd == 0)
		{
			// The event is passed as a HANDLE, where 0 would mean 'no event'
			const int fd = ::fcntl(_eventFd, F_DUPFD_CLOEXEC, 1);
			::close(_eventFd);
			_eventFd = fd;
		}
		if (_eventFd >= 0)
			_event = reinterpret_cast<HANDLE>(static_cast<intptr_t>(_eventFd));
#endif

		VclEnsure(_d3d12Fence, "GPU event created");
		VclEnsure(_event, "Host event created");
	}

	Semaphore::~Semaphore()
	{
		// Release the fence first: it may still signal the event for a wait that timed out
		_d3d12Fence.Reset();

#ifdef _WIN32
		if (_event)
			::CloseHandle(_event);
#else
		if (_eventFd >= 0)
			::close(_eventFd);
#endif
	}

	uint64_t Semaphore::signal(ID3D12CommandQueue* queue)
	{
		uint64_t new_value = ++_value;
		VCL_DIRECT3D_SAFE_CALL(queue->Signal(_d3d12Fence.Get(), new_value));

		return new_value;
	}

	bool Semaphore::wait(uint64_t value, std::chrono::milliseconds duration)
	{
		using namespace std::chrono;
		using clock = steady_clock;

		const bool infinite = duration == Infinite;
		const auto deadline = infinite ? clock::time_point::max() : clock::now() + duration;

		// The event may have been set by an earlier, timed-out wait for a lower
		// value, so check the fence again after every wake-up
		while (_d3d12Fence->GetCompletedValue() < value)
		{
			auto remaining = Infinite;
			if (!infinite)
			{
				const auto left = deadline - clock::now();
				if (left <= clock::duration::zero())
					return false;

				// Round up, so that the wait doesn't end before the deadline
				remaining = duration_cast<milliseconds>(left);
				if (remaining < left)
					remaining += milliseconds{ 1 };
			}

			VCL_DIRECT3D_SAFE_CALL(_d3d12Fence->SetEventOnCompletion(value, _event));
			// A timeout loops back to the deadline check: Windows may end the wait up
			// to a timer tick early. A failed wait returns, so it doesn't busy-loop.
			if (waitForEvent(remaining) == WaitResult::Failed)
				return false;
		}

		return true;
	}

#ifdef _WIN32
	void Semaphore::wait(uint64_t value, HANDLE additional_event, std::chrono::milliseconds duration)
	{
		// Wait for the fence with the re-checking wait above: '_event' may still be
		// set by an earlier wait that timed out
		const auto start = std::chrono::steady_clock::now();
		::WaitForSingleObject(additional_event, toTimeout(duration));
		if (duration == Infinite)
		{
			wait(value);
		} else
		{
			const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
			wait(value, std::max(duration - elapsed, std::chrono::milliseconds::zero()));
		}
	}
#endif

	bool Semaphore::sync(ID3D12CommandQueue* queue, std::chrono::milliseconds duration)
	{
		return wait(signal(queue), duration);
	}

	Semaphore::WaitResult Semaphore::waitForEvent(std::chrono::milliseconds duration)
	{
#ifdef _WIN32
		switch (::WaitForSingleObject(_event, toTimeout(duration)))
		{
		case WAIT_OBJECT_0: return WaitResult::Signaled;
		case WAIT_TIMEOUT: return WaitResult::TimedOut;
		default: return WaitResult::Failed;
		}
#else
		const int timeout = duration == Infinite ? -1 : static_cast<int>(std::min<long long>(duration.count(), INT_MAX));

		pollfd fd = { _eventFd, POLLIN, 0 };
		int result;
		do
		{
			result = ::poll(&fd, 1, timeout);
		} while (result < 0 && errno == EINTR);

		if (result == 0)
			return WaitResult::TimedOut;
		if (result < 0 || (fd.revents & (POLLERR | POLLNVAL)) != 0)
			return WaitResult::Failed;

		// Reset the event
		eventfd_t count;
		return eventfd_read(_eventFd, &count) == 0 ? WaitResult::Signaled : WaitResult::Failed;
#endif
	}
}}}
