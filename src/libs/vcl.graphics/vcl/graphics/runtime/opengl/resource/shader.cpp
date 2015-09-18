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

// Header
#include <vcl/graphics/runtime/opengl/resource/shader.h>

#ifdef VCL_OPENGL_SUPPORT

#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Shader::Shader(ShaderType type, int tag, const char* source)
	: Runtime::Shader(type, tag)
	{
		// Create the shader object
		_shaderObject = glCreateShader(toGLenum(type));
		glShaderSource(_shaderObject, 1, &source, nullptr);
		glCompileShader(_shaderObject);

		AssertBlock
		{
			printInfoLog();
		}

		Ensure(_shaderObject > 0, "Shader is created");
	}

	Shader::~Shader()
	{
		if (_shaderObject)
			glDeleteShader(_shaderObject);
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
		char* info_log;

		glGetShaderiv(_shaderObject, GL_INFO_LOG_LENGTH, &info_log_length);

		if (info_log_length > 1)
		{
			info_log = new char[info_log_length];
			glGetShaderInfoLog(_shaderObject, info_log_length, &chars_written, info_log);
			printf("%s\n", info_log);
			delete[] info_log;
		}
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
