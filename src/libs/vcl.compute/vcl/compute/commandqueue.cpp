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
#include <vcl/compute/commandqueue.h>

// C++ standard library

// VCL

namespace Vcl { namespace Compute
{
	CommandQueue::CommandQueue(CommandQueue&&)
	{

	}
	CommandQueue& CommandQueue::operator = (CommandQueue&&)
	{
		return *this;
	}

	void CommandQueue::setZero(BufferView dst)
	{
		unsigned int pattern = 0;
		fill(dst, &pattern, sizeof(pattern));
	}

	void CommandQueue::setZero(ref_ptr<Buffer> dst)
	{
		setZero(BufferView{ dst });
	}

	void CommandQueue::copy(ref_ptr<Buffer> dst, ref_ptr<const Buffer> src)
	{
		copy(BufferView{ dst }, ConstBufferView{ src });
	}

	void CommandQueue::read(void* dst, ref_ptr<const Buffer> src, bool blocking)
	{
		read(dst, ConstBufferView{ src }, blocking);
	}
	void CommandQueue::write(ref_ptr<Buffer> dst, void* src, bool blocking)
	{
		write(BufferView{ dst }, src, blocking);
	}
	void CommandQueue::fill(ref_ptr<Buffer> dst, const void* pattern, size_t pattern_size)
	{
		fill(BufferView{ dst }, pattern, pattern_size);
	}
}}
