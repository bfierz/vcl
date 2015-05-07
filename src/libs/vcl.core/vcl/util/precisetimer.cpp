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
#include <vcl/util/precisetimer.h>

#include <vcl/core/contract.h>

namespace Vcl { namespace Util
{

#ifdef VCL_ABI_POSIX
	timespec diff(timespec start, timespec end)
	{
		timespec temp;
		if ((end.tv_nsec-start.tv_nsec)<0) {
			temp.tv_sec = end.tv_sec-start.tv_sec-1;
			temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
		} else {
			temp.tv_sec = end.tv_sec-start.tv_sec;
			temp.tv_nsec = end.tv_nsec-start.tv_nsec;
		}
		return temp;
	}
#endif
	void PreciseTimer::start()
	{
#ifdef VCL_ABI_WINAPI
		QueryPerformanceCounter(&mStartTime);
#elif defined VCL_ABI_POSIX
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mStartTime);
#endif // VCL_ABI_WINAPI
	}

	void PreciseTimer::stop()
	{
#ifdef VCL_ABI_WINAPI
		QueryPerformanceCounter(&mStopTime);
#elif defined VCL_ABI_POSIX
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mStopTime);
#endif // VCL_ABI_WINAPI
	}

	double PreciseTimer::interval(unsigned int nr_iterations) const
	{
		Require(nr_iterations > 0, "Number of iterations is at least 1.");

#ifdef VCL_ABI_WINAPI
		LARGE_INTEGER freq;
		if (QueryPerformanceFrequency(&freq) == false) return std::numeric_limits<double>::quiet_NaN();
		
		return ((double)(mStopTime.QuadPart - mStartTime.QuadPart) / (double) freq.QuadPart) / (double) nr_iterations;
#elif defined VCL_ABI_POSIX
		timespec thisdiff = diff(mStartTime, mStopTime);
		return (double(size_t(1e9)*thisdiff.tv_sec) + double(thisdiff.tv_nsec)) / (double)nr_iterations;
#endif // VCL_ABI_WINAPI
	}
}}
