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
#include <vcl/graphics/runtime/d3d12/resource/shader.h>

// C++ standard library
#include <regex>
#include <vector>

// C runtime
#include <cmath>
#include <cstring>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 {
	Shader::Shader(ShaderType type, int tag, const char* source, std::initializer_list<const char*> headers)
	: Runtime::Shader(type, tag)
	{
		// Create the shader object
		VclEnsure(!_compiled_shader.empty(), "Shader is created");
	}

	Shader::Shader
	(
		ShaderType type, int tag,
		stdext::span<const uint8_t> binary_data
	)
	: Runtime::Shader(type, tag)
	{
		_compiled_shader.assign(binary_data.begin(), binary_data.end());
		VclEnsure(!_compiled_shader.empty(), "Shader is created");
	}
}}}}
