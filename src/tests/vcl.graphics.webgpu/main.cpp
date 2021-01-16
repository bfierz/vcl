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

// VCL
#include <vcl/config/webgpu.h>

#ifndef VCL_ARCH_WEBASM
#include <dawn_native/DawnNative.h>
#include <dawn/dawn_proc.h>

std::unique_ptr<dawn_native::Instance> instance;

#else
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#endif

WGPUDevice device;

int main(int argc, char **argv)
{
#ifndef VCL_ARCH_WEBASM
	instance = std::make_unique<dawn_native::Instance>();
	instance->DiscoverDefaultAdapters();
	auto adapters = instance->GetAdapters();
	if (adapters.empty())
		return 1;

	// Run tests using CPU emulation
	for (auto& adapter : adapters)
	{
		if (adapter.GetDeviceType() == dawn_native::DeviceType::CPU)
		{
			device = adapter.CreateDevice();
			break;
		}
	}
	if (!device)
	{
		device = adapters[0].CreateDevice();
	}

	DawnProcTable procs = dawn_native::GetProcs();
	dawnProcSetProcs(&procs);
#else
	device = emscripten_webgpu_get_device();
	if (!device)
		return 1;
#endif

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
