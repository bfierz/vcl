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

// VCL
#include <vcl/graphics/d3d12/device.h>

// Google test
#include <gtest/gtest.h>

using Vcl::Graphics::D3D12::AdapterInfo;
using Vcl::Graphics::D3D12::DeviceType;
using Vcl::Graphics::D3D12::selectAdapter;

namespace {
	const AdapterInfo SmallGpu{ "Small GPU", true, 2ull << 30 };
	const AdapterInfo LargeGpu{ "Large GPU", true, 8ull << 30 };
	const AdapterInfo Software{ "Software", false, 0 };
}

TEST(D3D12AdapterSelection, HardwarePrefersMostDedicatedMemory)
{
	EXPECT_EQ(1, selectAdapter({ SmallGpu, LargeGpu, Software }, DeviceType::Hardware));
	EXPECT_EQ(0, selectAdapter({ LargeGpu, SmallGpu }, DeviceType::Hardware));
}

TEST(D3D12AdapterSelection, HardwareIgnoresSoftwareAdapters)
{
	EXPECT_EQ(1, selectAdapter({ Software, SmallGpu }, DeviceType::Hardware));
	EXPECT_EQ(-1, selectAdapter({ Software }, DeviceType::Hardware));
}

TEST(D3D12AdapterSelection, HardwareAcceptsAdaptersWithoutDedicatedMemory)
{
	// Integrated GPUs may report no dedicated memory
	const AdapterInfo integrated{ "Integrated GPU", true, 0 };
	EXPECT_EQ(0, selectAdapter({ integrated }, DeviceType::Hardware));
}

TEST(D3D12AdapterSelection, WarpSelectsSoftwareAdapter)
{
	EXPECT_EQ(1, selectAdapter({ LargeGpu, Software }, DeviceType::Warp));
	EXPECT_EQ(-1, selectAdapter({ LargeGpu }, DeviceType::Warp));
}

TEST(D3D12AdapterSelection, NoAdapters)
{
	EXPECT_EQ(-1, selectAdapter({}, DeviceType::Hardware));
	EXPECT_EQ(-1, selectAdapter({}, DeviceType::Warp));
}
