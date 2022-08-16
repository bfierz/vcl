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
#include <vcl/graphics/runtime/webgpu/resource/shader.h>

// C++ standard library
#include <regex>
#include <vector>

// C runtime
#include <cmath>
#include <cstring>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU {
	Shader::Shader(
		WGPUDevice device,
		ShaderType type,
		int tag,
		stdext::span<const uint8_t> binary_data)
	: Runtime::Shader(type, tag)
	{
		WGPUShaderModuleSPIRVDescriptor spirv_desc = {};
		spirv_desc.chain.sType = WGPUSType_ShaderModuleSPIRVDescriptor;
		spirv_desc.codeSize = static_cast<uint32_t>(binary_data.size()) / sizeof(uint32_t);
		spirv_desc.code = reinterpret_cast<const uint32_t*>(binary_data.data());

		WGPUShaderModuleDescriptor desc = {};
		desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&spirv_desc);

		_module = wgpuDeviceCreateShaderModule(device, &desc);
		VclEnsure(_module, "Shader is created");
	}
}}}}
