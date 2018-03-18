/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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

// C++ Standard Library

// Vulkan API
#include <vulkan/vulkan.h>

// Include the relevant parts from the library
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>

// Google test
#include <gtest/gtest.h>

// Tests the vulkan initialization
TEST(Vulkan, InitPlatformNoExtensions)
{
	using namespace Vcl::Graphics::Vulkan;

	// Initialize the Vulkan platform
	auto platform = std::make_unique<Platform>();
	VkInstance inst = *platform;

	// Verify the result
	EXPECT_TRUE(inst != nullptr) << "Platform not created.";

	for (int i = 0; i < platform->nrDevices(); i++)
	{
		auto& dev = platform->device(i);

		EXPECT_FALSE(dev.name().empty()) << "Physical device name is not valid.";
	}
}

// Tests the vulkan initialization
TEST(Vulkan, InitDevicesForAllPhysicalDevicesWithoutExtensions)
{
	using namespace Vcl::Graphics::Vulkan;

	// Initialize the Vulkan platform
	auto platform = std::make_unique<Platform>();

	// Create a context for each device
	for (int i = 0; i < platform->nrDevices(); i++)
	{
		auto& dev = platform->device(i);
		auto ctx = dev.createContext();
		VkDevice ptr = *ctx;

		EXPECT_TRUE(ptr != nullptr) << "Vulkan device is not created.";
	}
}

// Tests command queue allocation on a simple device
TEST(Vulkan, CommandQueueOnSimpleDevice)
{
	using namespace Vcl::Graphics::Vulkan;

	// Initialize the Vulkan platform
	auto platform = std::make_unique<Platform>();

	if (platform->nrDevices() == 0)
		return;

	auto& dev = platform->device(0);
	auto context = dev.createContext();

	CommandQueue queue{ context.get(), 0 };
	VkQueue gk_queue = queue;

	EXPECT_TRUE(gk_queue != nullptr) << "Vulkan command queue is not created.";
}