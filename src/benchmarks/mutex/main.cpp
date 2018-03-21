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

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <mutex>

// Boost library
#if defined(VCL_COMPILER_CLANG) && defined(VCL_ABI_WINAPI)
#	define BOOST_USE_WINDOWS_H
#	define BOOST_SP_USE_STD_ATOMIC
#endif

#include <boost/thread.hpp>

// Qt
#include <QtCore/QMutex>

// Win32
#ifdef VCL_ABI_WINAPI
#	include <Windows.h>
#endif

// Google benchmark
#include "benchmark/benchmark.h"

////////////////////////////////////////////////////////////////////////////////
// C++ STL Mutex
////////////////////////////////////////////////////////////////////////////////

// Mutex
std::mutex StdMutex;

static void BM_Std(benchmark::State& state)
{
	while (state.KeepRunning())
	{
		StdMutex.lock();
		StdMutex.unlock();
	}
}
// Register the function as a benchmark
BENCHMARK(BM_Std)->ThreadRange(1, 16);

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Boost Mutex
////////////////////////////////////////////////////////////////////////////////

// Mutex
boost::mutex BoostMutex;

static void BM_Boost(benchmark::State& state)
{
	while (state.KeepRunning())
	{
		BoostMutex.lock();
		BoostMutex.unlock();
	}
}
// Register the function as a benchmark
BENCHMARK(BM_Boost)->ThreadRange(1, 16);

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// C++ STL Mutex
////////////////////////////////////////////////////////////////////////////////

// Mutex
std::mutex QtMutex;

static void BM_Qt(benchmark::State& state)
{
	while (state.KeepRunning())
	{
		QtMutex.lock();
		QtMutex.unlock();
	}
}
// Register the function as a benchmark
BENCHMARK(BM_Qt)->ThreadRange(1, 16);

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Win32 API Critical Section
////////////////////////////////////////////////////////////////////////////////
#ifdef VCL_ABI_WINAPI

// Mutex
CRITICAL_SECTION CriticalSection;

static void BM_WindowsCriticalSection(benchmark::State& state)
{
	if (state.thread_index == 0)
	{
		InitializeCriticalSection(&CriticalSection);
	}

	while (state.KeepRunning())
	{
		EnterCriticalSection(&CriticalSection);
		LeaveCriticalSection(&CriticalSection);
	}

	if (state.thread_index == 0)
	{
		DeleteCriticalSection(&CriticalSection);
	}
}
// Register the function as a benchmark
BENCHMARK(BM_WindowsCriticalSection)->ThreadRange(1, 16);

#endif // VCL_ABI_WINAPI
////////////////////////////////////////////////////////////////////////////////

BENCHMARK_MAIN();
