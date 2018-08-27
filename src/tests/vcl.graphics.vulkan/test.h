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
#include <vector>

// Include the relevant parts from the library
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>

// Google test
#include <gtest/gtest.h>

class VulkanTest : public testing::Test
{
public:
	void SetUp() override
	{
		using namespace Vcl::Graphics::Vulkan;

		// Initialize the Vulkan platform
		_platform = std::make_unique<Platform>(stdext::make_span(_platform_extensions));
		auto& dev = _platform->device(0);
		_context = dev.createContext(stdext::make_span(_context_extensions));
	}

	void TearDown() override
	{
		_context.reset();
		_platform.reset();

		_context_extensions.clear();
		_platform_extensions.clear();
	}

protected:
	std::unique_ptr<Vcl::Graphics::Vulkan::Platform> _platform;
	std::unique_ptr<Vcl::Graphics::Vulkan::Context> _context;
		
	//! List of requested platform extensions
	std::vector<const char*> _platform_extensions;

	//! List of requested context extensions
	std::vector<const char*> _context_extensions;
};
