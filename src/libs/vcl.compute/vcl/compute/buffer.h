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
#include <functional>
#include <future>

// VCL
#include <vcl/core/memory/smart_ptr.h>

namespace Vcl { namespace Compute {
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
		ReadWrite = 0x3,

		//! Resource access is managed by the system
		Unified = 0x7
	};

	/*!
	 *	\brief Constant view on a part of a buffer
	 */
	class ConstBufferView
	{
	public:
		ConstBufferView(ref_ptr<const Buffer> buf);
		ConstBufferView(ref_ptr<const Buffer> buf, size_t offset, size_t size);

	public:
		size_t offset() const { return _offsetInBytes; }
		size_t size() const { return _sizeInBytes; }

	public:
		const Buffer& owner() const { return *_owner; }

	protected:
		//! Buffer to which the view belongs
		ref_ptr<Buffer> _owner;

		//! Offset in bytes
		size_t _offsetInBytes{ 0 };

		//! Size in bytes
		size_t _sizeInBytes{ 0 };
	};

	/*!
	 *	\brief View on a part of a buffer
	 */
	class BufferView : public ConstBufferView
	{
	public:
		BufferView(const ref_ptr<Buffer> buf);
		BufferView(const ref_ptr<Buffer> buf, size_t offset, size_t size);

	public:
		Buffer& owner() { return *_owner; }
	};

	/*!
	 *	\brief Abstraction for compute API linear memory buffers
	 */
	class Buffer
	{
	public:
		Buffer(BufferAccess hostAccess, size_t size);
		Buffer(const Buffer&) = delete;
		Buffer(Buffer&&);
		virtual ~Buffer() = default;

	public:
		Buffer& operator=(const Buffer&) = delete;
		Buffer& operator=(Buffer&&);

	public:
		BufferAccess hostAccess() const { return _hostAccess; }
		size_t size() const { return _sizeInBytes; }

	protected:
		void setSize(size_t size) { _sizeInBytes = size; }

	private:
		//! Host access to the buffers memory
		BufferAccess _hostAccess = BufferAccess::ReadWrite;

		//! Size in bytes
		size_t _sizeInBytes = 0;
	};
}}
