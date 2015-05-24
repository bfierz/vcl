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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <future>

// VCL

namespace Vcl { namespace Compute
{
	class Buffer;

	enum class BufferAccess
	{
		//! Resource is not accessible from the host
		None = 0x0,

		//! Resource has only write access from host
		Write = 0x1,

		//! Resource has only read access from host
		Read = 0x2,

		//! Resource has read and write access from host
		ReadWrite = 0x3
	};

	class BufferSyncPoint
	{
	public:
		BufferSyncPoint() = delete;
		BufferSyncPoint(const BufferSyncPoint& rhs) = delete;

		BufferSyncPoint(std::future<bool>&& future);
		BufferSyncPoint(std::future<bool>&& future, std::function<void()>&& callback);
		BufferSyncPoint(BufferSyncPoint&& rhs);
		~BufferSyncPoint();

		bool isReady() const;
		void sync() const;

	private:
		std::future<bool> _hasCompleted;
		std::function<void()> _onCompletion;
	};

	/*!
	 *	\brief View on a part of a buffer
	 */
	class BufferView
	{
	public:
		BufferView() = default;
		BufferView(int size, BufferAccess hostAccess, BufferAccess deviceAccess);
		virtual ~BufferView() = default;

	public:
		BufferAccess hostAccess() const { return _hostAccess; }
		BufferAccess deviceAccess() const { return _deviceAccess; }
		int offset() const { return _offsetInBytes; }
		int size() const { return _sizeInBytes; }

	public:
		virtual const Buffer& owner() const = 0;

	protected:
		//! Host access to the buffers memory
		BufferAccess _hostAccess = BufferAccess::ReadWrite;

		//! Device access to the buffers memory
		BufferAccess _deviceAccess = BufferAccess::ReadWrite;

		//! Offset in bytes
		int _offsetInBytes = 0;

		//! Size in bytes
		int _sizeInBytes = 0;
	};
	
	/*!
	 *	\brief Abstraction for compute API linear memory buffers
	 */
	class Buffer : public BufferView
	{
	public:
		Buffer(BufferAccess hostAccess, int size);
		Buffer(const Buffer&) = delete;
		Buffer(Buffer&&);
		virtual ~Buffer() = default;

	public:
		Buffer& operator =(const Buffer&) = delete;
		Buffer& operator =(Buffer&&);

	protected:
		virtual const Buffer& owner() const override;
	};
}}
