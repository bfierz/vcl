/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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

VCL_BEGIN_EXTERNAL_HEADERS
#ifdef VCL_ABI_WINAPI
#	include <windows.h>
#elif defined(VCL_ABI_POSIX)
#endif // VCL_ABI_WINAPI
VCL_END_EXTERNAL_HEADERS

// VCL
#include <vcl/core/3rdparty/format.h>

namespace Vcl { namespace Util
{
	void ReserveBottomMemory()
	{
#ifdef _WIN64
		static bool s_initialized = false;
		if ( s_initialized )
			return;
		s_initialized = true;
 
		// Start by reserving large blocks of address space, and then
		// gradually reduce the size in order to capture all of the
		// fragments. Technically we should continue down to 64 KB but
		// stopping at 1 MB is sufficient to keep most allocators out.
 
		const size_t LOW_MEM_LINE = 0x100000000LL;
		size_t totalReservation = 0;
		size_t numVAllocs = 0;
		size_t numHeapAllocs = 0;
		size_t oneMB = 1024 * 1024;
		for (size_t size = 256 * oneMB; size >= oneMB; size /= 2)
		{
			for (;;)
			{
				void* p = VirtualAlloc(0, size, MEM_RESERVE, PAGE_NOACCESS);
				if (!p)
					break;
 
				if ((size_t)p >= LOW_MEM_LINE)
				{
					// We don't need this memory, so release it completely.
					VirtualFree(p, 0, MEM_RELEASE);
					break;
				}
 
				totalReservation += size;
				++numVAllocs;
			}
		}
 
		// Now repeat the same process but making heap allocations, to use up
		// the already reserved heap blocks that are below the 4 GB line.
		HANDLE heap = GetProcessHeap();
		for (size_t blockSize = 64 * 1024; blockSize >= 16; blockSize /= 2)
		{
			for (;;)
			{
				void* p = HeapAlloc(heap, 0, blockSize);
				if (!p)
					break;
 
				if ((size_t)p >= LOW_MEM_LINE)
				{
					// We don't need this memory, so release it completely.
					HeapFree(heap, 0, p);
					break;
				}
 
				totalReservation += blockSize;
				++numHeapAllocs;
			}
		}
 
		// Perversely enough the CRT doesn't use the process heap. Suck up
		// the memory the CRT heap has already reserved.
		for (size_t blockSize = 64 * 1024; blockSize >= 16; blockSize /= 2)
		{
			for (;;)
			{
				void* p = malloc(blockSize);
				if (!p)
					break;
 
				if ((size_t)p >= LOW_MEM_LINE)
				{
					// We don't need this memory, so release it completely.
					free(p);
					break;
				}
 
				totalReservation += blockSize;
				++numHeapAllocs;
			}
		}
 
		// Print diagnostics showing how many allocations we had to make in
		// order to reserve all of low memory, typically less than 200.
		auto msgbuf = fmt::sprintf
		(
			"Reserved %1.3f MB (%d vallocs, %d heap allocs) of low-memory.\n",
			totalReservation / (1024 * 1024.0), (int) numVAllocs, (int) numHeapAllocs
		);

		OutputDebugStringA(msgbuf.c_str());
#endif
	}
}}
