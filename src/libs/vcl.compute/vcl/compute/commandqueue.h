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
#include <memory>
#include <string>
#include <unordered_map>

// VCL
namespace Vcl { namespace Compute { class BufferView; }}

namespace Vcl { namespace Compute
{
	class CommandQueue
	{
	public:
		CommandQueue(CommandQueue&&);
		CommandQueue(const CommandQueue&) = delete;

		CommandQueue& operator =(CommandQueue&&);
		CommandQueue& operator =(const CommandQueue&) = delete;

		virtual ~CommandQueue() = default;

	public:
		virtual void sync() = 0;

	public:
		virtual void read(void* dst, Vcl::Compute::BufferView& src, size_t offset, size_t size, bool blocking = false) = 0;
		virtual void write(Vcl::Compute::BufferView& dst, void* src, size_t offset, size_t size, bool blocking = false) = 0;

		virtual void fill(Vcl::Compute::BufferView& dst, const void* pattern, size_t pattern_size) = 0;
	};
}}
