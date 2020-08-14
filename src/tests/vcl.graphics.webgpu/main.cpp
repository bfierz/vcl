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

#include <dawn_native/DawnNative.h>
#include <dawn/dawn_proc.h>

std::unique_ptr<dawn_native::Instance> instance;
//WGPUInstance instance;
WGPUDevice device;

//void request_adapter_callback(WGPUAdapter received, void* userdata)
//{
//	*(WGPUAdapter*)userdata = received;
//}

int main(int argc, char **argv)
{
	//WGPUInstanceDescriptor inst_desc = {};
	//instance = wgpuCreateInstance(&inst_desc);

	instance = std::make_unique<dawn_native::Instance>();
	instance->DiscoverDefaultAdapters();
	dawn_native::Adapter adapter = instance->GetDefaultAdapter();
	device = adapter.CreateDevice();

	DawnProcTable procs = dawn_native::GetProcs();
	dawnProcSetProcs(&procs);

	/*WGPUAdapterId adapter = { 0 };
	wgpu_request_adapter_async(
		nullptr,
		2 | 4 | 8,
		request_adapter_callback,
		(void*)&adapter
	);
	WGPUDeviceDescriptor adapter_desc = {};
	adapter_desc.extensions.anisotropic_filtering = false;
	adapter_desc.limits.max_bind_groups = 1;
	device = wgpu_adapter_request_device(adapter, &adapter_desc, nullptr);*/

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
