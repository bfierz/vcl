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
#include <vcl/graphics/runtime/opengl/resource/shader.h>

// C++ standard library
#include <regex>
#include <vector>

// C runtime
#include <cstring>

// GSL
#include <string_span.h>

#ifdef VCL_OPENGL_SUPPORT

#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Shader::Shader(ShaderType type, int tag, const char* source, const char* header)
	: Runtime::Shader(type, tag)
	{
		Require(implies(type == ShaderType::ComputeShader, glewIsExtensionSupported("GL_ARB_compute_shader")), "Compute shaders are supported.");

		// Search for the version/extension block
		const char* version_begin = strstr(source, "#version");
		const char* version_end = version_begin;
		if (version_begin)
		{
			version_end = strchr(version_begin, '\n') + 1;
		}
		
		// Build the source table
		const char* table[3] =
		{
			version_begin != version_end ? version_begin : "",
			header ? header : "",
			version_begin != version_end ? version_end : source
		};
		GLint sizes[3] =
		{
			version_begin != version_end ? (version_end - version_begin) : 0,
			header ? strlen(header) : 0,
			strlen(source) - (version_begin != version_end ? (version_end - source) : 0)
		};

		// Create the shader object
		_glId = glCreateShader(toGLenum(type));
		glShaderSource(_glId, 3, table, sizes);
		glCompileShader(_glId);

		AssertBlock
		{
			printInfoLog();
		}

		Ensure(_glId > 0 && glIsShader(_glId), "Shader is created");
	}

	Shader::Shader(Shader&& rhs)
	: Resource(std::move(rhs))
	, Runtime::Shader(rhs)
	{
	}

	Shader::~Shader()
	{
		if (_glId)
			glDeleteShader(_glId);
	}

	GLenum Shader::toGLenum(ShaderType type)
	{
		switch (type)
		{
		case ShaderType::VertexShader:     return GL_VERTEX_SHADER;
		case ShaderType::ControlShader:    return GL_TESS_CONTROL_SHADER;
		case ShaderType::EvaluationShader: return GL_TESS_EVALUATION_SHADER;
		case ShaderType::GeometryShader:   return GL_GEOMETRY_SHADER;
		case ShaderType::FragmentShader:   return GL_FRAGMENT_SHADER;
		case ShaderType::ComputeShader:    return GL_COMPUTE_SHADER;
		default: { DebugError("Enumeration value is valid."); }
		}

		return GL_NONE;
	}

	void Shader::printInfoLog() const
	{
		int info_log_length = 0;
		int chars_written = 0;

		glGetShaderiv(_glId, GL_INFO_LOG_LENGTH, &info_log_length);

		if (info_log_length > 1)
		{
			std::vector<char> info_log(info_log_length);
			glGetShaderInfoLog(_glId, info_log_length, &chars_written, info_log.data());
			printf("%s\n", info_log.data());
		}
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
