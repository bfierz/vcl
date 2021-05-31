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
#include <vcl/core/handle.h>

// C++ standard library
#include <chrono>

namespace Vcl
{
	uint32_t createResourceHandleTag(void* owner) noexcept
	{
		// Generate a new resource tag
		const auto now = std::chrono::steady_clock::now();
		const size_t tagId = static_cast<size_t>(now.time_since_epoch().count()) ^ reinterpret_cast<size_t>(owner);

#if defined VCL_ARCH_X64
		return static_cast<uint32_t>(tagId) ^ static_cast<uint32_t>(tagId >> 32);
#elif defined VCL_ARCH_X86 || defined VCL_ARCH_ARM || defined VCL_ARCH_WEBASM
		return tagId;
#else
		VCL_ERROR("Unknown platform")
#endif
	}
}
