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

// C runtime
#include <cstring>

// C++ standard library
#include <string>

#ifdef VCL_ARCH_WEBASM
#	include <emscripten.h>
#	include <emscripten/html5.h>
#	include <emscripten/html5_webgpu.h>
#else
#	include <iostream>
#endif

WGPUDevice device;

#ifndef VCL_ARCH_WEBASM
namespace {
	WGPUInstance instance;

	template<typename Result>
	struct Request
	{
		Result Value{};
		bool Done{ false };
	};

	WGPUAdapter requestAdapter(bool fallback)
	{
		Request<WGPUAdapter> request;

		WGPURequestAdapterOptions options = {};
		options.backendType = WGPUBackendType_Undefined;
		options.forceFallbackAdapter = fallback;

		WGPURequestAdapterCallbackInfo callback = {};
		callback.mode = WGPUCallbackMode_WaitAnyOnly;
		callback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView, void* userdata1, void*) {
			auto* request = static_cast<Request<WGPUAdapter>*>(userdata1);
			request->Value = status == WGPURequestAdapterStatus_Success ? adapter : nullptr;
			request->Done = true;
		};
		callback.userdata1 = &request;

		WGPUFutureWaitInfo wait = { wgpuInstanceRequestAdapter(instance, &options, callback), false };
		wgpuInstanceWaitAny(instance, 1, &wait, UINT64_MAX);
		return request.Value;
	}

	WGPUDevice requestDevice(WGPUAdapter adapter)
	{
		Request<WGPUDevice> request;

		WGPUDeviceDescriptor desc = {};
		desc.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType type, WGPUStringView message, void*, void*) {
			std::cerr << "WebGPU error " << type << ": " << std::string(message.data, message.length == WGPU_STRLEN ? strlen(message.data) : message.length) << std::endl;
		};

		WGPURequestDeviceCallbackInfo callback = {};
		callback.mode = WGPUCallbackMode_WaitAnyOnly;
		callback.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView, void* userdata1, void*) {
			auto* request = static_cast<Request<WGPUDevice>*>(userdata1);
			request->Value = status == WGPURequestDeviceStatus_Success ? device : nullptr;
			request->Done = true;
		};
		callback.userdata1 = &request;

		WGPUFutureWaitInfo wait = { wgpuAdapterRequestDevice(adapter, &desc, callback), false };
		wgpuInstanceWaitAny(instance, 1, &wait, UINT64_MAX);
		return request.Value;
	}
}
#endif

int main(int argc, char** argv)
{
#ifndef VCL_ARCH_WEBASM
	// Waiting on futures requires the timed wait feature
	const WGPUInstanceFeatureName features[] = { WGPUInstanceFeatureName_TimedWaitAny };
	WGPUInstanceDescriptor instance_desc = {};
	instance_desc.requiredFeatureCount = 1;
	instance_desc.requiredFeatures = features;
	instance = wgpuCreateInstance(&instance_desc);
	if (!instance)
		return 1;

	// Run tests using CPU emulation when available
	WGPUAdapter adapter = requestAdapter(true);
	if (!adapter)
		adapter = requestAdapter(false);
	if (!adapter)
		return 1;

	device = requestDevice(adapter);
	if (!device)
		return 1;
#else
	device = emscripten_webgpu_get_device();
	if (!device)
		return 1;
#endif

	::testing::InitGoogleTest(&argc, argv);
	const int result = RUN_ALL_TESTS();

#ifndef VCL_ARCH_WEBASM
	wgpuDeviceRelease(device);
	wgpuAdapterRelease(adapter);
	wgpuInstanceRelease(instance);
#endif
	return result;
}
