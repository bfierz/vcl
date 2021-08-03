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
#include <cmath>
#include <cstring>

#ifdef VCL_OPENGL_SUPPORT

#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL {
	nonstd::expected<Shader, std::string> makeShader(ShaderType type, int tag, const char* source, std::initializer_list<const char*> headers)
	{
		Shader shader{ type, tag, source, headers };
		if (shader.checkCompilationState())
			return std::move(shader);
		else
			return nonstd::make_unexpected(shader.readInfoLog());
	}

	nonstd::expected<Shader, std::string> makeShader(ShaderType type, int tag, stdext::span<const uint8_t> binary_data, stdext::span<const unsigned int> spec_indices, stdext::span<const unsigned int> spec_values)
	{
		Shader shader{ type, tag, binary_data, spec_indices, spec_values };
		if (shader.checkCompilationState())
			return std::move(shader);
		else
			return nonstd::make_unexpected(shader.readInfoLog());
	}

	Shader::Shader(ShaderType type, int tag, const char* source, std::initializer_list<const char*> headers)
	: Runtime::Shader(type, tag)
	{
		VclRequire(implies(type == ShaderType::ComputeShader, glewIsExtensionSupported("GL_ARB_compute_shader")), "Compute shaders are supported.");

		// Search for the version/extension block
		const char* version_begin = strstr(source, "#version");
		const char* version_end = version_begin;
		if (version_begin)
		{
			version_end = strchr(version_begin, '\n') + 1;
		}

		// Build the source table
		std::vector<const char*> table;
		table.reserve(2 + headers.size());
		table.emplace_back(version_begin != version_end ? version_begin : "");
		for (auto header : headers)
			table.emplace_back(header);

		table.emplace_back(version_begin != version_end ? version_end : source);

		std::vector<GLint> sizes;
		sizes.reserve(2 + headers.size());
		sizes.emplace_back(version_begin != version_end ? static_cast<int>(version_end - version_begin) : 0);
		for (auto header : headers)
			sizes.emplace_back(header ? static_cast<int>(strlen(header)) : 0);

		sizes.emplace_back(static_cast<int>(strlen(source) - (version_begin != version_end ? (version_end - source) : 0)));

		// Create the shader object
		_glId = glCreateShader(toGLenum(type));
		glShaderSource(_glId, static_cast<GLsizei>(table.size()), table.data(), sizes.data());
		glCompileShader(_glId);

		VclEnsure(_glId > 0 && glIsShader(_glId), "Shader is created");
	}

	Shader::Shader(
		ShaderType type,
		int tag,
		stdext::span<const uint8_t> binary_data,
		stdext::span<const unsigned int> spec_indices,
		stdext::span<const unsigned int> spec_values)
	: Runtime::Shader(type, tag)
	{
		VclRequire(GLEW_ARB_gl_spirv, "SPIR-V is supported.");
		VclRequire(spec_indices.size() == spec_values.size(), "Specialization constants buffer have same size.");

		// Create the shader object
		_glId = glCreateShader(toGLenum(type));

		// Load the pre-compiled shader
		glShaderBinary(1, &_glId, GL_SHADER_BINARY_FORMAT_SPIR_V, binary_data.data(), static_cast<GLsizei>(binary_data.size()));

		// Specialize the shader to determine its final behaviour
		glSpecializeShaderARB(_glId, "main", static_cast<GLsizei>(spec_indices.size()), spec_indices.data(), spec_values.data());

		VclEnsure(_glId > 0 && glIsShader(_glId), "Shader is created");
	}

	Shader::Shader(Shader&& rhs)
	: Runtime::Shader(rhs)
	, Resource(std::move(rhs))
	{
	}

	Shader::~Shader()
	{
		if (_glId)
			glDeleteShader(_glId);
	}

	bool Shader::checkCompilationState() const
	{
		GLint compile_status = GL_TRUE;
		glGetShaderiv(_glId, GL_COMPILE_STATUS, &compile_status);
		return compile_status == GL_TRUE;
	}

	std::string Shader::readInfoLog() const
	{
		if (_glId == 0)
			return {};

		int info_log_length = 0;
		int chars_written = 0;

		glGetShaderiv(_glId, GL_INFO_LOG_LENGTH, &info_log_length);
		if (info_log_length > 1)
		{
			std::string info_log(info_log_length, '\0');
			glGetShaderInfoLog(_glId, info_log_length, &chars_written, const_cast<char*>(info_log.data()));
			return info_log;
		}
		return {};
	}

	bool Shader::isSpirvSupported()
	{
		return glewIsSupported("GL_ARB_gl_spirv") && glewIsSupported("GL_ARB_spirv_extensions");
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
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_NONE;
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
