/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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
#include <limits>

#ifdef VCL_STL_CHRONO
#	include <chrono>
#else
	VCL_BEGIN_EXTERNAL_HEADERS
#	ifdef VCL_ABI_WINAPI
#		include <windows.h>
#	elif defined(VCL_ABI_POSIX)
#		include <inttypes.h>
#		include <time.h>
#	endif
	VCL_END_EXTERNAL_HEADERS
#endif // VCL_STL_CHRONO

namespace Vcl { namespace Util
{
	class PreciseTimer
	{
	public:
		void start();
		void stop();

		double interval(unsigned int nr_iterations = 1) const;

	private:
#ifdef VCL_STL_CHRONO
		//! Time clock was started
		std::chrono::high_resolution_clock::time_point _startTime;

		//! Time clock was stopped
		std::chrono::high_resolution_clock::time_point _stopTime;
#elif defined(VCL_ABI_WINAPI)
		LARGE_INTEGER _startTime, _stopTime;
#elif defined(VCL_ABI_POSIX)
		timespec _startTime, _stopTime;
#endif // VCL_STL_CHRONO
	};
}}
