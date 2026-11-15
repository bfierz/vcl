/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2026 Basil Fierz
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

// C++ standard library
#include <chrono>

// VCL
#include <vcl/graphics/d3d12/device.h>
#include <vcl/graphics/d3d12/semaphore.h>

// Google test
#include <gtest/gtest.h>

extern std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

using namespace std::chrono_literals;
using Vcl::Graphics::D3D12::Semaphore;

TEST(D3D12Semaphore, SyncCompletes)
{
	Semaphore semaphore{ device->nativeDevice() };
	EXPECT_TRUE(semaphore.sync(device->defaultQueue()->nativeQueue(), 5000ms));
	EXPECT_EQ(1u, semaphore.nativeFence()->GetCompletedValue());
}

TEST(D3D12Semaphore, WaitForCompletedValueReturnsImmediately)
{
	Semaphore semaphore{ device->nativeDevice() };
	EXPECT_TRUE(semaphore.wait(0, 0ms));
}

TEST(D3D12Semaphore, WaitTimesOut)
{
	Semaphore semaphore{ device->nativeDevice() };

	const auto start = std::chrono::steady_clock::now();
	EXPECT_FALSE(semaphore.wait(1, 50ms));
	EXPECT_GE(std::chrono::steady_clock::now() - start, 50ms);
}

TEST(D3D12Semaphore, WaitReturnsWhenFenceIsSignaled)
{
	Semaphore semaphore{ device->nativeDevice() };
	ASSERT_FALSE(semaphore.wait(1, 0ms));

	semaphore.nativeFence()->Signal(1);
	EXPECT_TRUE(semaphore.wait(1, 5000ms));
}

TEST(D3D12Semaphore, EarlierWaitDoesNotEndLaterWait)
{
	Semaphore semaphore{ device->nativeDevice() };

	// The timed-out wait leaves the event registered for value 1
	EXPECT_FALSE(semaphore.wait(1, 10ms));
	semaphore.nativeFence()->Signal(1);

	// Signaling value 1 sets the event, but value 2 is not reached
	EXPECT_FALSE(semaphore.wait(2, 50ms));
	EXPECT_TRUE(semaphore.wait(1, 0ms));
}
