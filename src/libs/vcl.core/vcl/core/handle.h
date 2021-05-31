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

namespace Vcl
{
	uint32_t createResourceHandleTag(void* owner) noexcept;

	template<typename T>
	class Handle
	{
	public:
		Handle(int32_t idx = -1, uint32_t tag = 0)
		: _dataIdx(idx)
		, _tag(tag)
		{}
		
	public:
		int32_t index() const
		{
			return _dataIdx;
		}

		uint32_t tag() const
		{
			return _tag;
		}

		bool isValid() const
		{
			return _dataIdx > -1 && _tag != 0;
		}

	public:
		bool operator == (Handle<T> h) const
		{
			return _dataIdx == h._dataIdx && _tag == h.tag;
		}

		bool operator < (Handle<T> h) const
		{
			return _tag < h._tag || (_tag == h._tag && _dataIdx < h._dataIdx);
		}

	private:
		 int32_t _dataIdx;
		uint32_t _tag;
	};
}
