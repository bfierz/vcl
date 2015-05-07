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
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// Boost library
#include <boost/thread.hpp>

// Qt
#include <QtCore/QMutex>

// Win32
#ifdef VCL_ABI_WINAPI
#	include <Windows.h>
#endif

// VCL
#include <vcl/util/precisetimer.h>

// The loop counter
int MaxIterations;

// List of threads trying to obtain the mutexes
std::vector<std::thread> Threads;

// Locking all threads until all are ready
std::mutex m;
std::condition_variable halt;
bool ready = false;

// Various mutexes to test
std::mutex StdMutex;
boost::mutex BoostMutex;
QMutex QtMutex;

#ifdef VCL_ABI_WINAPI
CRITICAL_SECTION CriticalSection;
#endif

// Wait function blocking the threads until everything is setup
void wait(void)
{
	std::unique_lock<std::mutex> lk(m);
	halt.wait(lk, []{ return ready; });
}

// Functions testing mutexes
void stdMutex()
{
	wait();
	for (int i = 0; i < MaxIterations; i++)
	{
		StdMutex.lock();
		StdMutex.unlock();
	}
}

void boostMutex()
{
	wait();
	for (int i = 0; i < MaxIterations; i++)
	{
		BoostMutex.lock();
		BoostMutex.unlock();
	}
}

void qMutex()
{
	wait();
	for (int i = 0; i < MaxIterations; i++)
	{
		QtMutex.lock();
		QtMutex.unlock();
	}
}

#ifdef VCL_ABI_WINAPI
void winMutex()
{
	wait();
	for (int i = 0; i < MaxIterations; i++)
	{
		EnterCriticalSection(&CriticalSection);
		LeaveCriticalSection(&CriticalSection);
	}
}
#endif

typedef  void(*ThreadFunc)(void);

struct Job
{
	ThreadFunc Func;
	const char* Name;
};

#define JOB(FUNC) { FUNC, #FUNC }

int main(int argc, char* argv[])
{
	MaxIterations = 10000;

	Job jobs [] =
	{
		JOB(stdMutex),
		JOB(boostMutex),
		JOB(qMutex),
#ifdef VCL_ABI_WINAPI
		JOB(winMutex),
#endif
	};

	// Win32
#ifdef VCL_ABI_WINAPI
	InitializeCriticalSection(&CriticalSection);
#endif

	for (auto job : jobs)
	{
		// Block all threads
		ready = false;

		// Clear the threads from the previous run
		Threads.clear();

		// Queue the new tasks
		for (int i = 0; i < 10; i++)
		{
			Threads.emplace_back(job.Func);
		}

		// Start the threads
		Vcl::Util::PreciseTimer timer;
		timer.start();

		ready = true;
		halt.notify_all();

		for (auto& t : Threads)
		{
			t.join();
		}

		timer.stop();

		std::cout << "Time per mutex (" << job.Name << "): " << timer.interval() << std::endl;
	}

	// Win32
#ifdef VCL_ABI_WINAPI
	DeleteCriticalSection(&CriticalSection);
#endif
	return 0;
}
