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
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>

// C++ standard library
#include <vector>

// VCL
#include <vcl/core/contract.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	ProgramResources::ProgramResources(GLuint program)
	{
		Require(program > 0 && glIsProgram(program), "Shader program is defined.");
		Require(glewIsExtensionSupported("GL_ARB_program_interface_query"), "GL shader program interface is supported.");

		GLint nr_active_attributes = 0;
		GLint nr_active_outputs = 0;
		GLint nr_active_uniforms = 0;

		glGetProgramInterfaceiv(program, GL_PROGRAM_INPUT, GL_ACTIVE_RESOURCES, &nr_active_attributes);
		glGetProgramInterfaceiv(program, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES, &nr_active_outputs);
		glGetProgramInterfaceiv(program, GL_UNIFORM, GL_ACTIVE_RESOURCES, &nr_active_uniforms);

		// Collect the information about the attributes
		if (nr_active_attributes > 0)
		{
			_attributes.reserve(nr_active_attributes);

			// Temporary for uniform names
			std::vector<char> name(32);

			const GLenum properties [] = { GL_LOCATION, GL_TYPE, GL_NAME_LENGTH };
			const size_t num_properties = sizeof(properties) / sizeof(GLenum);
			for (int u = 0; u < nr_active_attributes; ++u)
			{
				GLint values[num_properties];
				glGetProgramResourceiv(program, GL_PROGRAM_INPUT, u, num_properties, properties, num_properties, nullptr, values);
				
				// Read the uniform name
				name.resize(values[2]);
				glGetProgramResourceName(program, GL_PROGRAM_INPUT, u, (GLsizei) name.size(), nullptr, name.data());

				// Construct the uniform
				_attributes.emplace_back(AttributeData{ values[0], { name.begin(), name.end() - 1 }, toResourceType(values[1]) });
			}
		}

		// Collect the information about the attributes
		if (nr_active_outputs > 0)
		{
			_outputs.reserve(nr_active_outputs);

			// Temporary for uniform names
			std::vector<char> name(32);

			const GLenum properties [] = { GL_LOCATION, GL_TYPE, GL_NAME_LENGTH };
			const size_t num_properties = sizeof(properties) / sizeof(GLenum);
			for (int u = 0; u < nr_active_outputs; ++u)
			{
				GLint values[num_properties];
				glGetProgramResourceiv(program, GL_PROGRAM_OUTPUT, u, num_properties, properties, num_properties, nullptr, values);

				// Read the uniform name
				name.resize(values[2]);
				glGetProgramResourceName(program, GL_PROGRAM_OUTPUT, u, (GLsizei) name.size(), nullptr, name.data());

				// Construct the uniform
				_outputs.emplace_back(ProgramOutputData{ values[0], { name.begin(), name.end() - 1 }, toResourceType(values[1]) });
			}
		}

		// Collect the information about the uniforms
		if (nr_active_uniforms > 0)
		{
			_uniforms.reserve(nr_active_uniforms);

			// Temporary for uniform names
			std::vector<char> name(32);

			const GLenum properties [] = { GL_BLOCK_INDEX, GL_TYPE, GL_NAME_LENGTH, GL_LOCATION, GL_ARRAY_SIZE };
			const size_t num_properties = sizeof(properties) / sizeof(GLenum);
			for (int u = 0; u < nr_active_uniforms; ++u)
			{
				GLint values[num_properties];
				glGetProgramResourceiv(program, GL_UNIFORM, u, num_properties, properties, num_properties, nullptr, values);

				// Skip any uniforms that are in a block
				if (values[0] != -1)
					continue;

				// Read the uniform name
				name.resize(values[2]);
				glGetProgramResourceName(program, GL_UNIFORM, u, (GLsizei) name.size(), nullptr, name.data());
				
				// Construct the uniform
				_uniforms.emplace_back(UniformData{ values[3], { name.begin(), name.end() - 1 }, toResourceType(values[1]), values[4] });
			}
		}
	}
	
	GLenum ProgramResources::toGLenum(ProgramResourceType type)
	{
		switch (type)
		{
		case ProgramResourceType::Float : return GL_FLOAT     ;
		case ProgramResourceType::Float2: return GL_FLOAT_VEC2;
		case ProgramResourceType::Float3: return GL_FLOAT_VEC3;
		case ProgramResourceType::Float4: return GL_FLOAT_VEC4;

		case ProgramResourceType::Double : return GL_DOUBLE     ;
		case ProgramResourceType::Double2: return GL_DOUBLE_VEC2;
		case ProgramResourceType::Double3: return GL_DOUBLE_VEC3;
		case ProgramResourceType::Double4: return GL_DOUBLE_VEC4;

		case ProgramResourceType::Int : return GL_INT     ;
		case ProgramResourceType::Int2: return GL_INT_VEC2;
		case ProgramResourceType::Int3: return GL_INT_VEC3;
		case ProgramResourceType::Int4: return GL_INT_VEC4;
		/*
		case GL_UNSIGNED_INT     : return ProgramResourceType::UnsignedInt;
		case GL_UNSIGNED_INT_VEC2: return ProgramResourceType::UnsignedInt2;
		case GL_UNSIGNED_INT_VEC3: return ProgramResourceType::UnsignedInt3;
		case GL_UNSIGNED_INT_VEC4: return ProgramResourceType::UnsignedInt4;

		case GL_BOOL     : return ProgramResourceType::Bool;
		case GL_BOOL_VEC2: return ProgramResourceType::Bool2;
		case GL_BOOL_VEC3: return ProgramResourceType::Bool3;
		case GL_BOOL_VEC4: return ProgramResourceType::Bool4;

		case GL_FLOAT_MAT2:   return ProgramResourceType::FloatMatrix2;
		case GL_FLOAT_MAT2x3: return ProgramResourceType::FloatMatrix2x3;
		case GL_FLOAT_MAT2x4: return ProgramResourceType::FloatMatrix2x4;
		case GL_FLOAT_MAT3:   return ProgramResourceType::FloatMatrix3;
		case GL_FLOAT_MAT3x2: return ProgramResourceType::FloatMatrix3x2;
		case GL_FLOAT_MAT3x4: return ProgramResourceType::FloatMatrix3x4;
		case GL_FLOAT_MAT4:   return ProgramResourceType::FloatMatrix4;
		case GL_FLOAT_MAT4x2: return ProgramResourceType::FloatMatrix4x2;
		case GL_FLOAT_MAT4x3: return ProgramResourceType::FloatMatrix4x3;

		case GL_DOUBLE_MAT2:   return ProgramResourceType::DoubleMatrix2;
		case GL_DOUBLE_MAT2x3: return ProgramResourceType::DoubleMatrix2x3;
		case GL_DOUBLE_MAT2x4: return ProgramResourceType::DoubleMatrix2x4;
		case GL_DOUBLE_MAT3:   return ProgramResourceType::DoubleMatrix3;
		case GL_DOUBLE_MAT3x2: return ProgramResourceType::DoubleMatrix3x2;
		case GL_DOUBLE_MAT3x4: return ProgramResourceType::DoubleMatrix3x4;
		case GL_DOUBLE_MAT4:   return ProgramResourceType::DoubleMatrix4;
		case GL_DOUBLE_MAT4x2: return ProgramResourceType::DoubleMatrix4x2;
		case GL_DOUBLE_MAT4x3: return ProgramResourceType::DoubleMatrix4x3;

		case GL_SAMPLER_1D:				 return ProgramResourceType::Sampler1D;
		case GL_SAMPLER_1D_ARRAY:		 return ProgramResourceType::Sampler1DArray;
		case GL_SAMPLER_1D_ARRAY_SHADOW: return ProgramResourceType::Sampler1DArrayShadow;
		case GL_SAMPLER_1D_SHADOW:		 return ProgramResourceType::Sampler1DShadow;
		case GL_SAMPLER_2D:				 return ProgramResourceType::Sampler2D;
		case GL_SAMPLER_2D_ARRAY:		 return ProgramResourceType::Sampler2DArray;
		case GL_SAMPLER_2D_ARRAY_SHADOW: return ProgramResourceType::Sampler2DArrayShadow;
		case GL_SAMPLER_2D_MULTISAMPLE:	 return ProgramResourceType::Sampler2DMultisample;
		case GL_SAMPLER_2D_RECT:		 return ProgramResourceType::Sampler2DRect;
		case GL_SAMPLER_2D_RECT_SHADOW:	 return ProgramResourceType::Sampler2DRectShadow;
		case GL_SAMPLER_2D_SHADOW:		 return ProgramResourceType::Sampler2DShadow;

		case GL_SAMPLER_3D:					   return ProgramResourceType::Sampler3D;
		case GL_SAMPLER_BUFFER:				   return ProgramResourceType::SamplerBuffer;
		case GL_SAMPLER_CUBE:				   return ProgramResourceType::SamplerCube;
		case GL_SAMPLER_CUBE_MAP_ARRAY:		   return ProgramResourceType::SamplerCubeArray;
		case GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW: return ProgramResourceType::SamplerCubeArrayShadow;
		case GL_SAMPLER_CUBE_SHADOW:		   return ProgramResourceType::SamplerCubeShadow;

		case GL_INT_SAMPLER_1D:				return ProgramResourceType::IntSampler1D;
		case GL_INT_SAMPLER_1D_ARRAY:		return ProgramResourceType::IntSampler1DArray;
		case GL_INT_SAMPLER_2D:				return ProgramResourceType::IntSampler2D;
		case GL_INT_SAMPLER_2D_ARRAY:		return ProgramResourceType::IntSampler2DArray;
		case GL_INT_SAMPLER_2D_MULTISAMPLE: return ProgramResourceType::IntSampler2DMultisample;
		case GL_INT_SAMPLER_2D_RECT:		return ProgramResourceType::IntSampler2DRect;
		case GL_INT_SAMPLER_3D:				return ProgramResourceType::IntSampler3D;
		case GL_INT_SAMPLER_BUFFER:			return ProgramResourceType::IntSamplerBuffer;
		case GL_INT_SAMPLER_CUBE:			return ProgramResourceType::IntSamplerCube;
		case GL_INT_SAMPLER_CUBE_MAP_ARRAY: return ProgramResourceType::IntSamplerCubeArray;
				
		case GL_UNSIGNED_INT_SAMPLER_1D:			 return ProgramResourceType::UnsignedIntSampler1D;
		case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:		 return ProgramResourceType::UnsignedIntSampler1DArray;
		case GL_UNSIGNED_INT_SAMPLER_2D:			 return ProgramResourceType::UnsignedIntSampler2D;
		case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:		 return ProgramResourceType::UnsignedIntSampler2DArray;
		case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE: return ProgramResourceType::UnsignedIntSampler2DMultisample;
		case GL_UNSIGNED_INT_SAMPLER_2D_RECT:		 return ProgramResourceType::UnsignedIntSampler2DRect;
		case GL_UNSIGNED_INT_SAMPLER_3D:			 return ProgramResourceType::UnsignedIntSampler3D;
		case GL_UNSIGNED_INT_SAMPLER_BUFFER:		 return ProgramResourceType::UnsignedIntSamplerBuffer;
		case GL_UNSIGNED_INT_SAMPLER_CUBE:			 return ProgramResourceType::UnsignedIntSamplerCube;
		case GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY: return ProgramResourceType::UnsignedIntSamplerCubeArray;
		*/
		default:
			DebugError("Unrecognized Type");
		}

		return GL_NONE;
	}
	ProgramResourceType ProgramResources::toResourceType(GLenum type)
	{
		switch (type)
		{
		case GL_FLOAT     : return ProgramResourceType::Float;
		case GL_FLOAT_VEC2: return ProgramResourceType::Float2;
		case GL_FLOAT_VEC3: return ProgramResourceType::Float3;
		case GL_FLOAT_VEC4: return ProgramResourceType::Float4;

		case GL_DOUBLE:      return ProgramResourceType::Double;
		case GL_DOUBLE_VEC2: return ProgramResourceType::Double2;
		case GL_DOUBLE_VEC3: return ProgramResourceType::Double3;
		case GL_DOUBLE_VEC4: return ProgramResourceType::Double4;

		case GL_INT     : return ProgramResourceType::Int;
		case GL_INT_VEC2: return ProgramResourceType::Int2;
		case GL_INT_VEC3: return ProgramResourceType::Int3;
		case GL_INT_VEC4: return ProgramResourceType::Int4;

		case GL_UNSIGNED_INT     : return ProgramResourceType::UnsignedInt;
		case GL_UNSIGNED_INT_VEC2: return ProgramResourceType::UnsignedInt2;
		case GL_UNSIGNED_INT_VEC3: return ProgramResourceType::UnsignedInt3;
		case GL_UNSIGNED_INT_VEC4: return ProgramResourceType::UnsignedInt4;

		case GL_BOOL     : return ProgramResourceType::Bool;
		case GL_BOOL_VEC2: return ProgramResourceType::Bool2;
		case GL_BOOL_VEC3: return ProgramResourceType::Bool3;
		case GL_BOOL_VEC4: return ProgramResourceType::Bool4;

		case GL_FLOAT_MAT2:   return ProgramResourceType::FloatMatrix2;
		case GL_FLOAT_MAT2x3: return ProgramResourceType::FloatMatrix2x3;
		case GL_FLOAT_MAT2x4: return ProgramResourceType::FloatMatrix2x4;
		case GL_FLOAT_MAT3:   return ProgramResourceType::FloatMatrix3;
		case GL_FLOAT_MAT3x2: return ProgramResourceType::FloatMatrix3x2;
		case GL_FLOAT_MAT3x4: return ProgramResourceType::FloatMatrix3x4;
		case GL_FLOAT_MAT4:   return ProgramResourceType::FloatMatrix4;
		case GL_FLOAT_MAT4x2: return ProgramResourceType::FloatMatrix4x2;
		case GL_FLOAT_MAT4x3: return ProgramResourceType::FloatMatrix4x3;

		case GL_DOUBLE_MAT2:   return ProgramResourceType::DoubleMatrix2;
		case GL_DOUBLE_MAT2x3: return ProgramResourceType::DoubleMatrix2x3;
		case GL_DOUBLE_MAT2x4: return ProgramResourceType::DoubleMatrix2x4;
		case GL_DOUBLE_MAT3:   return ProgramResourceType::DoubleMatrix3;
		case GL_DOUBLE_MAT3x2: return ProgramResourceType::DoubleMatrix3x2;
		case GL_DOUBLE_MAT3x4: return ProgramResourceType::DoubleMatrix3x4;
		case GL_DOUBLE_MAT4:   return ProgramResourceType::DoubleMatrix4;
		case GL_DOUBLE_MAT4x2: return ProgramResourceType::DoubleMatrix4x2;
		case GL_DOUBLE_MAT4x3: return ProgramResourceType::DoubleMatrix4x3;

		case GL_SAMPLER_1D:				 return ProgramResourceType::Sampler1D;
		case GL_SAMPLER_1D_ARRAY:		 return ProgramResourceType::Sampler1DArray;
		case GL_SAMPLER_1D_ARRAY_SHADOW: return ProgramResourceType::Sampler1DArrayShadow;
		case GL_SAMPLER_1D_SHADOW:		 return ProgramResourceType::Sampler1DShadow;
		case GL_SAMPLER_2D:				 return ProgramResourceType::Sampler2D;
		case GL_SAMPLER_2D_ARRAY:		 return ProgramResourceType::Sampler2DArray;
		case GL_SAMPLER_2D_ARRAY_SHADOW: return ProgramResourceType::Sampler2DArrayShadow;
		case GL_SAMPLER_2D_MULTISAMPLE:	 return ProgramResourceType::Sampler2DMultisample;
		case GL_SAMPLER_2D_RECT:		 return ProgramResourceType::Sampler2DRect;
		case GL_SAMPLER_2D_RECT_SHADOW:	 return ProgramResourceType::Sampler2DRectShadow;
		case GL_SAMPLER_2D_SHADOW:		 return ProgramResourceType::Sampler2DShadow;

		case GL_SAMPLER_3D:					   return ProgramResourceType::Sampler3D;
		case GL_SAMPLER_BUFFER:				   return ProgramResourceType::SamplerBuffer;
		case GL_SAMPLER_CUBE:				   return ProgramResourceType::SamplerCube;
		case GL_SAMPLER_CUBE_MAP_ARRAY:		   return ProgramResourceType::SamplerCubeArray;
		case GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW: return ProgramResourceType::SamplerCubeArrayShadow;
		case GL_SAMPLER_CUBE_SHADOW:		   return ProgramResourceType::SamplerCubeShadow;

		case GL_INT_SAMPLER_1D:				return ProgramResourceType::IntSampler1D;
		case GL_INT_SAMPLER_1D_ARRAY:		return ProgramResourceType::IntSampler1DArray;
		case GL_INT_SAMPLER_2D:				return ProgramResourceType::IntSampler2D;
		case GL_INT_SAMPLER_2D_ARRAY:		return ProgramResourceType::IntSampler2DArray;
		case GL_INT_SAMPLER_2D_MULTISAMPLE: return ProgramResourceType::IntSampler2DMultisample;
		case GL_INT_SAMPLER_2D_RECT:		return ProgramResourceType::IntSampler2DRect;
		case GL_INT_SAMPLER_3D:				return ProgramResourceType::IntSampler3D;
		case GL_INT_SAMPLER_BUFFER:			return ProgramResourceType::IntSamplerBuffer;
		case GL_INT_SAMPLER_CUBE:			return ProgramResourceType::IntSamplerCube;
		case GL_INT_SAMPLER_CUBE_MAP_ARRAY: return ProgramResourceType::IntSamplerCubeArray;
				
		case GL_UNSIGNED_INT_SAMPLER_1D:			 return ProgramResourceType::UnsignedIntSampler1D;
		case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:		 return ProgramResourceType::UnsignedIntSampler1DArray;
		case GL_UNSIGNED_INT_SAMPLER_2D:			 return ProgramResourceType::UnsignedIntSampler2D;
		case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:		 return ProgramResourceType::UnsignedIntSampler2DArray;
		case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE: return ProgramResourceType::UnsignedIntSampler2DMultisample;
		case GL_UNSIGNED_INT_SAMPLER_2D_RECT:		 return ProgramResourceType::UnsignedIntSampler2DRect;
		case GL_UNSIGNED_INT_SAMPLER_3D:			 return ProgramResourceType::UnsignedIntSampler3D;
		case GL_UNSIGNED_INT_SAMPLER_BUFFER:		 return ProgramResourceType::UnsignedIntSamplerBuffer;
		case GL_UNSIGNED_INT_SAMPLER_CUBE:			 return ProgramResourceType::UnsignedIntSamplerCube;
		case GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY: return ProgramResourceType::UnsignedIntSamplerCubeArray;
				
		default:
			DebugError("Unrecognized Type");
		}

		return ProgramResourceType::Invalid;
	}

	const char* ProgramResources::name(GLenum type)
	{
		switch (type)
		{
		case GL_FLOAT:		return "float";
		case GL_FLOAT_VEC2:	return "vec2";
		case GL_FLOAT_VEC3:	return "vec3";
		case GL_FLOAT_VEC4:	return "vec4";

		case GL_FLOAT_MAT2: return "mat2";
		case GL_FLOAT_MAT3: return "mat3";
		case GL_FLOAT_MAT4: return "mat4";

		case GL_FLOAT_MAT2x3: return "mat2x3";
		case GL_FLOAT_MAT2x4: return "mat2x4";
		case GL_FLOAT_MAT3x2: return "mat3x2";
		case GL_FLOAT_MAT3x4: return "mat3x4";
		case GL_FLOAT_MAT4x2: return "mat4x2";
		case GL_FLOAT_MAT4x3: return "mat4x3";

		case GL_INT:		return "int";
		case GL_INT_VEC2:	return "ivec2";
		case GL_INT_VEC3:	return "ivec3";
		case GL_INT_VEC4:	return "ivec4";

		case GL_UNSIGNED_INT:		return "uint";
		case GL_UNSIGNED_INT_VEC2:	return "uivec2";
		case GL_UNSIGNED_INT_VEC3:	return "uivec3";
		case GL_UNSIGNED_INT_VEC4:	return "uivec4";
		default: { DebugError("Enumeration value is valid."); }
		}

		return "";
	}

	ShaderProgram::ShaderProgram(const ShaderProgramDescription& desc)
	{
		Require
		(
			implies
			(
				desc.ComputeShader, 
				!(desc.VertexShader || desc.TessControlShader || desc.TessEvalShader || desc.GeometryShader || desc.FragmentShader)
			),
			"Compute shader is not combined with graphics shaders."
		);
		Require(implies(!desc.ComputeShader, desc.VertexShader && desc.FragmentShader), "VertexShader and Fragment shader are set.");

		// Create a program for the shader
		_glId = glCreateProgram();

		if (desc.ComputeShader)
			glAttachShader(_glId, desc.ComputeShader->id());
		else
		{
			if (desc.VertexShader)
				glAttachShader(_glId, desc.VertexShader->id());

			if (desc.TessControlShader)
				glAttachShader(_glId, desc.TessControlShader->id());

			if (desc.TessEvalShader)
				glAttachShader(_glId, desc.TessEvalShader->id());

			if (desc.GeometryShader)
				glAttachShader(_glId, desc.GeometryShader->id());

			if (desc.FragmentShader)
				glAttachShader(_glId, desc.FragmentShader->id());
		}

		// Link the program
		glLinkProgram(id());

		// Collect the uniforms of this program
		_resources = std::make_unique<ProgramResources>(id());

		AssertBlock
		{
			printInfoLog();
		}
		Ensure(id() > 0, "Shader program is created");
	}

	void ShaderProgram::bind()
	{
		Require(id() > 0, "Shader program is created");

		glUseProgram(id());
	}

	void ShaderProgram::printInfoLog() const
	{
		int info_log_length = 0;
		int chars_written = 0;

		if (_glId == 0)
			return;

		glGetProgramiv(_glId, GL_INFO_LOG_LENGTH, &info_log_length);

		if (info_log_length > 1)
		{
			std::vector<char> info_log(info_log_length);
			glGetProgramInfoLog(_glId, info_log_length, &chars_written, info_log.data());
			printf("%s\n", info_log.data());
		}
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
