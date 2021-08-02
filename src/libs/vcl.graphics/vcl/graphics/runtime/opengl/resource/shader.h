/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

#ifdef VCL_OPENGL_SUPPORT

// C++ standard library
#include <initializer_list>

// VCL
#include <vcl/core/3rdparty/expected.hpp>
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/opengl/resource/resource.h>
#include <vcl/graphics/runtime/resource/shader.h>
#include <vcl/graphics/opengl/gl.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL {
	class Shader : public Runtime::Shader, public Resource
	{
	public:
		Shader(ShaderType type, int tag, const char* source, std::initializer_list<const char*> headers = {});
		Shader(ShaderType type, int tag,
			stdext::span<const uint8_t> binary_data,
			stdext::span<const unsigned int> spec_indices = {},
			stdext::span<const unsigned int> spec_values = {});
		Shader(Shader&& rhs);
		virtual ~Shader();

		//! Access the shader's compilation state
		//! \returns The shader's compilation state
		bool checkCompilationState() const;

		//! Access the shader log
		//! \returns The shader log
		std::string readInfoLog() const;

		static bool isSpirvSupported();
		static GLenum toGLenum(ShaderType type);
	};

	/// Create a new shader from a source string
	/// @returns The compiled shader, or the error-string in case of failure
	nonstd::expected<Shader, std::string> makeShader(ShaderType type, int tag, const char* source, std::initializer_list<const char*> headers = {});

	/// Create a new shader from a SPIR-V binary
	/// @returns The compiled shader, or the error-string in case of failure
	nonstd::expected<Shader, std::string> makeShader(ShaderType type, int tag, stdext::span<const uint8_t> binary_data, stdext::span<const unsigned int> spec_indices = {}, stdext::span<const unsigned int> spec_values = {});
}}}}
#endif // VCL_OPENGL_SUPPORT
