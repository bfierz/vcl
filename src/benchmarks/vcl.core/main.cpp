/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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

// Relevant libraries to test
#include <vector>

VCL_BEGIN_EXTERNAL_HEADERS
#include <foonathan/memory/container.hpp>
#include <foonathan/memory/namespace_alias.hpp>
#include <foonathan/memory/memory_resource_adapter.hpp>
#include <foonathan/memory/std_allocator.hpp>
#include <foonathan/memory/temporary_allocator.hpp>

// Google benchmark
#include "benchmark/benchmark.h"
VCL_END_EXTERNAL_HEADERS

// VCL
#include <vcl/core/memory/allocator.h>

template<class T>
using StdTempAllocator = memory::std_allocator<T, memory::temporary_allocator>;
template<class T>
using NoMutexStdTempAllocator = memory::std_allocator<T, memory::temporary_allocator, memory::no_mutex>;

template<class T>
using StdPmrAllocator = memory::std_allocator<T, memory::memory_resource_allocator>;

class OpaqueObject
{
	int _someMember{ 1 };
};

const int kMemorySize = 512;

void BM_InitTempThreadSafeAllocator(benchmark::State& state)
{
	memory::temporary_stack stack{4096};
	while (state.KeepRunning())
	{
		memory::temporary_allocator alloc(stack);
		std::vector<OpaqueObject, StdTempAllocator<OpaqueObject>> vec(alloc);
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_InitTempAllocator(benchmark::State& state)
{
	memory::temporary_stack stack{ 4096 };
	while (state.KeepRunning())
	{
		memory::temporary_allocator alloc(stack);
		std::vector<OpaqueObject, NoMutexStdTempAllocator<OpaqueObject>> vec(alloc);
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_PodTempThreadSafeAllocator(benchmark::State& state)
{
	memory::temporary_stack stack{ 4096 };
	while (state.KeepRunning())
	{
		memory::temporary_allocator alloc(stack);
		std::vector<int, StdTempAllocator<int>> vec(alloc);
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_PodTempAllocator(benchmark::State& state)
{
	memory::temporary_stack stack{ 4096 };
	while (state.KeepRunning())
	{
		memory::temporary_allocator alloc(stack);
		std::vector<int, NoMutexStdTempAllocator<int>> vec(alloc);
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_PodPmrAllocator(benchmark::State& state)
{
	using namespace Vcl::Core;

	memory::new_allocator new_alloc;
	memory::memory_resource_adapter<memory::new_allocator> resource(std::move(new_alloc));

	while (state.KeepRunning())
	{
		StdPmrAllocator<int> alloc(&resource);
		std::vector<int, StdPmrAllocator<int>> vec(alloc);
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_InitPmrAllocator(benchmark::State& state)
{
	using namespace Vcl::Core;

	memory::new_allocator new_alloc;
	memory::memory_resource_adapter<memory::new_allocator> resource(std::move(new_alloc));
	while (state.KeepRunning())
	{
		StdPmrAllocator<OpaqueObject> alloc(&resource);
		std::vector<OpaqueObject, StdPmrAllocator<OpaqueObject>> vec(alloc);
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}
void BM_PodStdAllocator(benchmark::State& state)
{
	using namespace Vcl::Core;

	while (state.KeepRunning())
	{
		std::vector<int> vec;
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_InitStdAllocator(benchmark::State& state)
{
	while (state.KeepRunning())
	{
		std::vector<OpaqueObject> vec;
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_InitCustomAllocator(benchmark::State& state)
{
	using namespace Vcl::Core;

	while (state.KeepRunning())
	{
		std::vector<int, Allocator<int, StandardAllocPolicy<int>, ObjectTraits<int>>> vec;
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_PodNoInitCustomAllocator(benchmark::State& state)
{
	using namespace Vcl::Core;

	while (state.KeepRunning())
	{
		std::vector<int, Allocator<int, StandardAllocPolicy<int>, NoInitObjectTraits<int>>> vec;
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

void BM_NoInitCustomAllocator(benchmark::State& state)
{
	using namespace Vcl::Core;

	while (state.KeepRunning())
	{
		std::vector<OpaqueObject, Allocator<OpaqueObject, StandardAllocPolicy<OpaqueObject>, NoInitObjectTraits<OpaqueObject>>> vec;
		vec.resize(kMemorySize);
		benchmark::DoNotOptimize(vec.size());
	}
}

// Register the function as a benchmark
BENCHMARK(BM_InitTempThreadSafeAllocator);
BENCHMARK(BM_InitTempAllocator);
BENCHMARK(BM_PodTempThreadSafeAllocator);
BENCHMARK(BM_PodTempAllocator);

BENCHMARK(BM_PodPmrAllocator);
BENCHMARK(BM_InitPmrAllocator);

BENCHMARK(BM_PodStdAllocator);
BENCHMARK(BM_InitStdAllocator);

BENCHMARK(BM_InitCustomAllocator);
BENCHMARK(BM_PodNoInitCustomAllocator);
BENCHMARK(BM_NoInitCustomAllocator);

BENCHMARK_MAIN();
