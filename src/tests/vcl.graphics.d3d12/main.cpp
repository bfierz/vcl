/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
#include <gtest/gtest.h>

// C++ standard library
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

// VCL
#include <vcl/graphics/d3d12/device.h>

std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

//! Creates the device only when tests run, so that listing them needs no GPU
class DeviceEnvironment : public ::testing::Environment
{
public:
	void SetUp() override
	{
		using namespace Vcl::Graphics::D3D12;

		if (!isRuntimeAvailable())
			GTEST_SKIP() << "No D3D12 runtime (on Linux, WSL provides libd3d12.so and libdxcore.so)";

			// WARP by default on Windows. Wine (vkd3d-proton) and WSL provide no WARP
			// adapter, so other platforms default to a hardware adapter.
			// 'VCL_D3D12_DEVICE=hardware' or 'VCL_D3D12_DEVICE=warp' overrides the default.
#ifdef _WIN32
		DeviceType type = DeviceType::Warp;
#else
		DeviceType type = DeviceType::Hardware;
#endif
		if (const char* device_type = std::getenv("VCL_D3D12_DEVICE"))
		{
			if (std::strcmp(device_type, "hardware") == 0)
				type = DeviceType::Hardware;
			else if (std::strcmp(device_type, "warp") == 0)
				type = DeviceType::Warp;
		}

		try
		{
			device = std::make_unique<Device>(type);
		} catch (const std::exception& e)
		{
			FAIL() << e.what();
		}
		std::cout << "Running tests on graphics adapter: " << device->adapterName() << std::endl;
	}

	void TearDown() override
	{
		device.reset();
	}
};

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new DeviceEnvironment);
	return RUN_ALL_TESTS();
}
