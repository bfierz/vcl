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
	ConstBufferView::ConstBufferView(ref_ptr<const Buffer> buf)
	: ConstBufferView(buf, 0, buf->size())
	{

	}

	ConstBufferView::ConstBufferView(ref_ptr<const Buffer> buf, size_t offset, size_t size)
	: _owner(const_pointer_cast<Buffer>(buf))
	, _offsetInBytes(offset)
	, _sizeInBytes(size)
	{

	}
	
	BufferView::BufferView(ref_ptr<Buffer> buf)
	: ConstBufferView(buf)
	{

	}

	BufferView::BufferView(ref_ptr<Buffer> buf, size_t offset, size_t size)
	: ConstBufferView(buf, offset, size)
	{

	}

	Buffer::Buffer(BufferAccess hostAccess, size_t size)
	: _hostAccess(hostAccess)
	, _sizeInBytes(size)

	{

	}
}}
