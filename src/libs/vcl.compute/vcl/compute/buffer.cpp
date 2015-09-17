/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#include <vcl/compute/buffer.h>

namespace Vcl { namespace Compute
{
	BufferSyncPoint::BufferSyncPoint(std::future<bool>&& future)
	: _hasCompleted(std::move(future))
	{
	}

	BufferSyncPoint::BufferSyncPoint(std::future<bool>&& future, std::function<void()>&& callback)
	: _hasCompleted(std::move(future))
	, _onCompletion(callback)
	{
	}

	BufferSyncPoint::BufferSyncPoint(BufferSyncPoint&& rhs)
	{
		_hasCompleted = std::move(rhs._hasCompleted);
		_onCompletion = std::move(rhs._onCompletion);
	}

	BufferSyncPoint::~BufferSyncPoint()
	{
	}

	bool BufferSyncPoint::isReady() const
	{
		return _hasCompleted.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	}

	void BufferSyncPoint::sync() const
	{
		_hasCompleted.wait();
		if (_onCompletion)
			_onCompletion();
	}

	BufferView::BufferView(int size, BufferAccess hostAccess, BufferAccess deviceAccess)
	: _sizeInBytes(size)
	, _hostAccess(hostAccess)
	, _deviceAccess(deviceAccess)
	{

	}

	Buffer::Buffer(BufferAccess hostAccess, int size)
	: BufferView(size, hostAccess, BufferAccess::ReadWrite)
	{
	}

	const Buffer& Buffer::owner() const
	{
		return *this;
	}
}}
