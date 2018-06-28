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

VCL_BEGIN_EXTERNAL_HEADERS
// FMT
#	include <fmt/format.h>
VCL_END_EXTERNAL_HEADERS

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/opengl/gl.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/resource/texture.h>
#include <vcl/graphics/runtime/opengl/state/sampler.h>

#ifdef VCL_OPENGL_SUPPORT

#define VCL_TYPE_TO_ENUM(type, val, name) case ProgramResourceType::type: return val;
#define VCL_ENUM_TO_TYPE(type, val, name) case val: return ProgramResourceType::type;
#define VCL_ENUM_TO_NAME(type, val, name) case val: return name;

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	ProgramAttributes::ProgramAttributes(GLuint program)
	{
		VclRequire(program > 0 && glIsProgram(program), "Shader program is defined.");
		VclRequire(glewIsExtensionSupported("GL_ARB_program_interface_query"), "GL shader program interface is supported.");

		GLint nr_active_attributes = 0;
		glGetProgramInterfaceiv(program, GL_PROGRAM_INPUT, GL_ACTIVE_RESOURCES, &nr_active_attributes);

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
				_attributes.emplace_back(AttributeData{ values[0], { name.begin(), name.end() - 1 }, ProgramResources::toResourceType(values[1]) });
			}
		}
	}

	ProgramOutput::ProgramOutput(GLuint program)
	{
		VclRequire(program > 0 && glIsProgram(program), "Shader program is defined.");
		VclRequire(glewIsExtensionSupported("GL_ARB_program_interface_query"), "GL shader program interface is supported.");

		GLint nr_active_outputs = 0;
		glGetProgramInterfaceiv(program, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES, &nr_active_outputs);

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
				_outputs.emplace_back(ProgramOutputData{ values[0], { name.begin(), name.end() - 1 }, ProgramResources::toResourceType(values[1]) });
			}
		}
	}

	ProgramUniforms::ProgramUniforms(GLuint program)
	{
		VclRequire(program > 0 && glIsProgram(program), "Shader program is defined.");
		VclRequire(glewIsExtensionSupported("GL_ARB_program_interface_query"), "GL shader program interface is supported.");

		GLint nr_active_uniforms = 0;
		glGetProgramInterfaceiv(program, GL_UNIFORM, GL_ACTIVE_RESOURCES, &nr_active_uniforms);

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

				// Concert the resource type
				ProgramResourceType type = ProgramResources::toResourceType(values[1]);

				GLint resLoc = -1;
				if (type >= ProgramResourceType::Sampler1D)
				{
					resLoc = glGetProgramResourceLocation(program, GL_UNIFORM, name.data());
					glProgramUniform1i(program, values[3], resLoc);
				}

				// Construct the uniform
				_uniforms.emplace_back(UniformData{ values[3], { name.begin(), name.end() - 1 }, type, values[4], resLoc });
			}
		}
	}

	ProgramUniformBlocks::ProgramUniformBlocks(GLuint program)
	{
		VclRequire(program > 0 && glIsProgram(program), "Shader program is defined.");
		VclRequire(glewIsExtensionSupported("GL_ARB_program_interface_query"), "GL shader program interface is supported.");

		GLint nr_active_uniforms = 0;
		glGetProgramInterfaceiv(program, GL_UNIFORM_BLOCK, GL_ACTIVE_RESOURCES, &nr_active_uniforms);

		// Collect the information about the uniforms
		if (nr_active_uniforms > 0)
		{
			_resources.reserve(nr_active_uniforms);

			// Temporary for uniform names
			std::vector<char> name(32);

			const GLenum properties [] = { GL_NAME_LENGTH };
			const size_t num_properties = sizeof(properties) / sizeof(GLenum);
			for (int u = 0; u < nr_active_uniforms; ++u)
			{
				GLint values[num_properties];
				glGetProgramResourceiv(program, GL_UNIFORM_BLOCK, u, num_properties, properties, num_properties, nullptr, values);

				// Read the uniform name
				name.resize(values[0]);
				glGetProgramResourceName(program, GL_UNIFORM_BLOCK, u, (GLsizei) name.size(), nullptr, name.data());

				GLint loc = glGetProgramResourceIndex(program, GL_UNIFORM_BLOCK, name.data());
				GLint res_loc = -1;
				glGetActiveUniformBlockiv(program, loc, GL_UNIFORM_BLOCK_BINDING, &res_loc);

				// Construct the uniform
				_resources.emplace_back(loc, std::string{ name.begin(), name.end() - 1 }, res_loc);
			}
		}
	}

	ProgramBuffers::ProgramBuffers(GLuint program)
	{
		VclRequire(program > 0 && glIsProgram(program), "Shader program is defined.");
		VclRequire(glewIsExtensionSupported("GL_ARB_program_interface_query"), "GL shader program interface is supported.");

		GLint nr_active_uniforms = 0;
		glGetProgramInterfaceiv(program, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &nr_active_uniforms);

		// Collect the information about the uniforms
		if (nr_active_uniforms > 0)
		{
			_resources.reserve(nr_active_uniforms);

			// Temporary for uniform names
			std::vector<char> name(32);

			const GLenum properties [] = { GL_NAME_LENGTH, GL_BUFFER_BINDING };
			const size_t num_properties = sizeof(properties) / sizeof(GLenum);
			for (int u = 0; u < nr_active_uniforms; ++u)
			{
				GLint values[num_properties];
				glGetProgramResourceiv(program, GL_SHADER_STORAGE_BLOCK, u, num_properties, properties, num_properties, nullptr, values);

				// Read the uniform name
				name.resize(values[0]);
				glGetProgramResourceName(program, GL_SHADER_STORAGE_BLOCK, u, (GLsizei) name.size(), nullptr, name.data());

				GLint loc = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, name.data());
				GLint res_loc = values[1];

				// Construct the uniform
				_resources.emplace_back(loc, std::string{ name.begin(), name.end() - 1 }, res_loc);
			}
		}
	}

	ProgramResources::ProgramResources(GLuint program)
	: _attributes(program)
	, _outputs(program)
	, _uniforms(program)
	, _uniformBlocks(program)
	, _buffers(program)
	{
	}
	
	GLenum ProgramResources::toGLenum(ProgramResourceType type)
	{
		switch (type)
		{
		VCL_TYPE_TO_ENUM(Float,                GL_FLOAT,                   "float")
		VCL_TYPE_TO_ENUM(Float2,               GL_FLOAT_VEC2,              "vec2")
		VCL_TYPE_TO_ENUM(Float3,               GL_FLOAT_VEC3,              "vec3")
		VCL_TYPE_TO_ENUM(Float4,               GL_FLOAT_VEC4,              "vec4")
		VCL_TYPE_TO_ENUM(Double,               GL_DOUBLE,                  "double")
		VCL_TYPE_TO_ENUM(Double2,              GL_DOUBLE_VEC2,             "dvec2")
		VCL_TYPE_TO_ENUM(Double3,              GL_DOUBLE_VEC3,             "dvec3")
		VCL_TYPE_TO_ENUM(Double4,              GL_DOUBLE_VEC4,             "dvec4")
		VCL_TYPE_TO_ENUM(Int,                  GL_INT,                     "int")
		VCL_TYPE_TO_ENUM(Int2,                 GL_INT_VEC2,                "ivec2")
		VCL_TYPE_TO_ENUM(Int3,                 GL_INT_VEC3,                "ivec3")
		VCL_TYPE_TO_ENUM(Int4,                 GL_INT_VEC4,                "ivec4")
		VCL_TYPE_TO_ENUM(UnsignedInt,          GL_UNSIGNED_INT,            "uint")
		VCL_TYPE_TO_ENUM(UnsignedInt2,         GL_UNSIGNED_INT_VEC2,       "uvec2")
		VCL_TYPE_TO_ENUM(UnsignedInt3,         GL_UNSIGNED_INT_VEC3,       "uvec3")
		VCL_TYPE_TO_ENUM(UnsignedInt4,         GL_UNSIGNED_INT_VEC4,       "uvec4")
		VCL_TYPE_TO_ENUM(Bool,                 GL_BOOL,                    "bool")
		VCL_TYPE_TO_ENUM(Bool2,                GL_BOOL_VEC2,               "bvec2")
		VCL_TYPE_TO_ENUM(Bool3,                GL_BOOL_VEC3,               "bvec3")
		VCL_TYPE_TO_ENUM(Bool4,                GL_BOOL_VEC4,               "bvec4")
		VCL_TYPE_TO_ENUM(FloatMatrix2,         GL_FLOAT_MAT2,              "mat2")
		VCL_TYPE_TO_ENUM(FloatMatrix2x3,       GL_FLOAT_MAT2x3,            "mat2x3")
		VCL_TYPE_TO_ENUM(FloatMatrix2x4,       GL_FLOAT_MAT2x4,            "mat2x4")
		VCL_TYPE_TO_ENUM(FloatMatrix3,         GL_FLOAT_MAT3,              "mat3")
		VCL_TYPE_TO_ENUM(FloatMatrix3x2,       GL_FLOAT_MAT3x2,            "mat3x2")
		VCL_TYPE_TO_ENUM(FloatMatrix3x4,       GL_FLOAT_MAT3x4,            "mat3x4")
		VCL_TYPE_TO_ENUM(FloatMatrix4,         GL_FLOAT_MAT4,              "mat4")
		VCL_TYPE_TO_ENUM(FloatMatrix4x2,       GL_FLOAT_MAT4x2,            "mat4x2")
		VCL_TYPE_TO_ENUM(FloatMatrix4x3,       GL_FLOAT_MAT4x3,            "mat4x3")
		VCL_TYPE_TO_ENUM(DoubleMatrix2,        GL_DOUBLE_MAT2,             "dmat2")
		VCL_TYPE_TO_ENUM(DoubleMatrix2x3,      GL_DOUBLE_MAT2x3,           "dmat2x3")
		VCL_TYPE_TO_ENUM(DoubleMatrix2x4,      GL_DOUBLE_MAT2x4,           "dmat2x4")
		VCL_TYPE_TO_ENUM(DoubleMatrix3,        GL_DOUBLE_MAT3,             "dmat3")
		VCL_TYPE_TO_ENUM(DoubleMatrix3x2,      GL_DOUBLE_MAT3x2,           "dmat3x2")
		VCL_TYPE_TO_ENUM(DoubleMatrix3x4,      GL_DOUBLE_MAT3x4,           "dmat3x4")
		VCL_TYPE_TO_ENUM(DoubleMatrix4,        GL_DOUBLE_MAT4,             "dmat4")
		VCL_TYPE_TO_ENUM(DoubleMatrix4x2,      GL_DOUBLE_MAT4x2,           "dmat4x2")
		VCL_TYPE_TO_ENUM(DoubleMatrix4x3,      GL_DOUBLE_MAT4x3,           "dmat4x3")

		VCL_TYPE_TO_ENUM(Sampler1D,                       GL_SAMPLER_1D,                          "sampler1D")
		VCL_TYPE_TO_ENUM(Sampler1DArray,                  GL_SAMPLER_1D_ARRAY,                    "sampler1DArray")
		VCL_TYPE_TO_ENUM(Sampler1DArrayShadow,            GL_SAMPLER_1D_ARRAY_SHADOW,             "sampler1DArrayShadow")
		VCL_TYPE_TO_ENUM(Sampler1DShadow,                 GL_SAMPLER_1D_SHADOW,                   "sampler1DShadow")
		VCL_TYPE_TO_ENUM(Sampler2D,                       GL_SAMPLER_2D,                          "sampler2D")
		VCL_TYPE_TO_ENUM(Sampler2DArray,                  GL_SAMPLER_2D_ARRAY,                    "sampler2DArray")
		VCL_TYPE_TO_ENUM(Sampler2DArrayShadow,            GL_SAMPLER_2D_ARRAY_SHADOW,             "sampler2DArrayShadow")
		VCL_TYPE_TO_ENUM(Sampler2DMultisample,            GL_SAMPLER_2D_MULTISAMPLE,              "sampler2DMultisample")
		VCL_TYPE_TO_ENUM(Sampler2DRect,                   GL_SAMPLER_2D_RECT,                     "sampler2DRect")
		VCL_TYPE_TO_ENUM(Sampler2DRectShadow,             GL_SAMPLER_2D_RECT_SHADOW,              "sampler2DRectShadow")
		VCL_TYPE_TO_ENUM(Sampler2DShadow,                 GL_SAMPLER_2D_SHADOW,                   "sampler2DShadow")
		VCL_TYPE_TO_ENUM(Sampler3D,                       GL_SAMPLER_3D,                          "sampler3D")
		VCL_TYPE_TO_ENUM(SamplerBuffer,                   GL_SAMPLER_BUFFER,                      "samplerBuffer")
		VCL_TYPE_TO_ENUM(SamplerCube,                     GL_SAMPLER_CUBE,                        "samplerCube")
		VCL_TYPE_TO_ENUM(SamplerCubeArray,                GL_SAMPLER_CUBE_MAP_ARRAY,              "samplerCubeArray")
		VCL_TYPE_TO_ENUM(SamplerCubeArrayShadow,          GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW,       "samplerCubeArrayShadow")
		VCL_TYPE_TO_ENUM(SamplerCubeShadow,               GL_SAMPLER_CUBE_SHADOW,                 "samplerCubeShadow")
		VCL_TYPE_TO_ENUM(IntSampler1D,                    GL_INT_SAMPLER_1D,                      "isampler1D")
		VCL_TYPE_TO_ENUM(IntSampler1DArray,               GL_INT_SAMPLER_1D_ARRAY,                "isampler1DArray")
		VCL_TYPE_TO_ENUM(IntSampler2D,                    GL_INT_SAMPLER_2D,                      "isampler2D")
		VCL_TYPE_TO_ENUM(IntSampler2DArray,               GL_INT_SAMPLER_2D_ARRAY,                "isampler2DArray")
		VCL_TYPE_TO_ENUM(IntSampler2DMultisample,         GL_INT_SAMPLER_2D_MULTISAMPLE,          "isampler2DMultisample")
		VCL_TYPE_TO_ENUM(IntSampler2DRect,                GL_INT_SAMPLER_2D_RECT,                 "isampler2DRect")
		VCL_TYPE_TO_ENUM(IntSampler3D,                    GL_INT_SAMPLER_3D,                      "isampler3D")
		VCL_TYPE_TO_ENUM(IntSamplerBuffer,                GL_INT_SAMPLER_BUFFER,                  "isamplerBuffer")
		VCL_TYPE_TO_ENUM(IntSamplerCube,                  GL_INT_SAMPLER_CUBE,                    "isamplerCube")
		VCL_TYPE_TO_ENUM(IntSamplerCubeArray,             GL_INT_SAMPLER_CUBE_MAP_ARRAY,          "isamplerCubeArray")
		VCL_TYPE_TO_ENUM(UnsignedIntSampler1D,            GL_UNSIGNED_INT_SAMPLER_1D,             "usampler1D")
		VCL_TYPE_TO_ENUM(UnsignedIntSampler1DArray,       GL_UNSIGNED_INT_SAMPLER_1D_ARRAY,       "usampler1DArray")
		VCL_TYPE_TO_ENUM(UnsignedIntSampler2D,            GL_UNSIGNED_INT_SAMPLER_2D,             "usampler2D")
		VCL_TYPE_TO_ENUM(UnsignedIntSampler2DArray,       GL_UNSIGNED_INT_SAMPLER_2D_ARRAY,       "usampler2DArray")
		VCL_TYPE_TO_ENUM(UnsignedIntSampler2DMultisample, GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE, "usampler2DMultisample")
		VCL_TYPE_TO_ENUM(UnsignedIntSampler2DRect,        GL_UNSIGNED_INT_SAMPLER_2D_RECT,        "usampler2DRect")
		VCL_TYPE_TO_ENUM(UnsignedIntSampler3D,            GL_UNSIGNED_INT_SAMPLER_3D,             "usampler3D")
		VCL_TYPE_TO_ENUM(UnsignedIntSamplerBuffer,        GL_UNSIGNED_INT_SAMPLER_BUFFER,         "usamplerBuffer")
		VCL_TYPE_TO_ENUM(UnsignedIntSamplerCube,          GL_UNSIGNED_INT_SAMPLER_CUBE,           "usamplerCube")
		VCL_TYPE_TO_ENUM(UnsignedIntSamplerCubeArray,     GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY, "usamplerCubeArray")
		
		VCL_TYPE_TO_ENUM(Image1D,                   GL_IMAGE_1D,                                "image1D")
		VCL_TYPE_TO_ENUM(Image2D,                   GL_IMAGE_2D,                                "image2D")
		VCL_TYPE_TO_ENUM(Image3D,                   GL_IMAGE_3D,                                "image3D")
		VCL_TYPE_TO_ENUM(ImageCube,                 GL_IMAGE_CUBE,                              "imageCube")
		VCL_TYPE_TO_ENUM(Image2DRect,               GL_IMAGE_2D_RECT,                           "image2DRect")
		VCL_TYPE_TO_ENUM(Image1DArray,              GL_IMAGE_1D_ARRAY,                          "image1DArray")
		VCL_TYPE_TO_ENUM(Image2DArray,              GL_IMAGE_2D_ARRAY,                          "image2DArray")
		VCL_TYPE_TO_ENUM(ImageCubeArray,            GL_IMAGE_CUBE_MAP_ARRAY,                    "image2DCubeArray")
		VCL_TYPE_TO_ENUM(ImageBuffer,               GL_IMAGE_BUFFER,                            "imageBuffer")
		VCL_TYPE_TO_ENUM(Image2DMS,                 GL_IMAGE_2D_MULTISAMPLE,                    "image2DMS")
		VCL_TYPE_TO_ENUM(Image2DMSArray,            GL_IMAGE_2D_MULTISAMPLE_ARRAY,              "image2DMSArray")
		VCL_TYPE_TO_ENUM(IntImage1D,                GL_INT_IMAGE_1D,                            "iimage1D")
		VCL_TYPE_TO_ENUM(IntImage2D,                GL_INT_IMAGE_2D,                            "iimage2D")
		VCL_TYPE_TO_ENUM(IntImage3D,                GL_INT_IMAGE_3D,                            "iimage3D")
		VCL_TYPE_TO_ENUM(IntImageCube,              GL_INT_IMAGE_CUBE,                          "iimageCube")
		VCL_TYPE_TO_ENUM(IntImage2DRect,            GL_INT_IMAGE_2D_RECT,                       "iimage2DRect")
		VCL_TYPE_TO_ENUM(IntImage1DArray,           GL_INT_IMAGE_1D_ARRAY,                      "iimage1DArray")
		VCL_TYPE_TO_ENUM(IntImage2DArray,           GL_INT_IMAGE_2D_ARRAY,                      "iimage2DArray")
		VCL_TYPE_TO_ENUM(IntImageCubeArray,         GL_INT_IMAGE_CUBE_MAP_ARRAY,                "iimage2DCubeArray")
		VCL_TYPE_TO_ENUM(IntImageBuffer,            GL_INT_IMAGE_BUFFER,                        "iimageBuffer")
		VCL_TYPE_TO_ENUM(IntImage2DMS,              GL_INT_IMAGE_2D_MULTISAMPLE,                "iimage2DMS")
		VCL_TYPE_TO_ENUM(IntImage2DMSArray,         GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY,          "iimage2DMSArray")
		VCL_TYPE_TO_ENUM(UnsignedIntImage1D,        GL_UNSIGNED_INT_IMAGE_1D,                   "uimage1D")
		VCL_TYPE_TO_ENUM(UnsignedIntImage2D,        GL_UNSIGNED_INT_IMAGE_2D,                   "uimage2D")
		VCL_TYPE_TO_ENUM(UnsignedIntImage3D,        GL_UNSIGNED_INT_IMAGE_3D,                   "uimage3D")
		VCL_TYPE_TO_ENUM(UnsignedIntImageCube,      GL_UNSIGNED_INT_IMAGE_CUBE,                 "uimageCube")
		VCL_TYPE_TO_ENUM(UnsignedIntImage2DRect,    GL_UNSIGNED_INT_IMAGE_2D_RECT,              "uimage2DRect")
		VCL_TYPE_TO_ENUM(UnsignedIntImage1DArray,   GL_UNSIGNED_INT_IMAGE_1D_ARRAY,             "uimage1DArray")
		VCL_TYPE_TO_ENUM(UnsignedIntImage2DArray,   GL_UNSIGNED_INT_IMAGE_2D_ARRAY,             "uimage2DArray")
		VCL_TYPE_TO_ENUM(UnsignedIntImageCubeArray, GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY,       "uimage2DCubeArray")
		VCL_TYPE_TO_ENUM(UnsignedIntImageBuffer,    GL_UNSIGNED_INT_IMAGE_BUFFER,               "uimageBuffer")
		VCL_TYPE_TO_ENUM(UnsignedIntImage2DMS,      GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE,       "uimage2DMS")
		VCL_TYPE_TO_ENUM(UnsignedIntImage2DMSArray, GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY, "uimage2DMSArray")

		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_NONE;
	}
	ProgramResourceType ProgramResources::toResourceType(GLenum type)
	{
		switch (type)
		{
		VCL_ENUM_TO_TYPE(Float,                GL_FLOAT,                   "float")
		VCL_ENUM_TO_TYPE(Float2,               GL_FLOAT_VEC2,              "vec2")
		VCL_ENUM_TO_TYPE(Float3,               GL_FLOAT_VEC3,              "vec3")
		VCL_ENUM_TO_TYPE(Float4,               GL_FLOAT_VEC4,              "vec4")
		VCL_ENUM_TO_TYPE(Double,               GL_DOUBLE,                  "double")
		VCL_ENUM_TO_TYPE(Double2,              GL_DOUBLE_VEC2,             "dvec2")
		VCL_ENUM_TO_TYPE(Double3,              GL_DOUBLE_VEC3,             "dvec3")
		VCL_ENUM_TO_TYPE(Double4,              GL_DOUBLE_VEC4,             "dvec4")
		VCL_ENUM_TO_TYPE(Int,                  GL_INT,                     "int")
		VCL_ENUM_TO_TYPE(Int2,                 GL_INT_VEC2,                "ivec2")
		VCL_ENUM_TO_TYPE(Int3,                 GL_INT_VEC3,                "ivec3")
		VCL_ENUM_TO_TYPE(Int4,                 GL_INT_VEC4,                "ivec4")
		VCL_ENUM_TO_TYPE(UnsignedInt,          GL_UNSIGNED_INT,            "uint")
		VCL_ENUM_TO_TYPE(UnsignedInt2,         GL_UNSIGNED_INT_VEC2,       "uvec2")
		VCL_ENUM_TO_TYPE(UnsignedInt3,         GL_UNSIGNED_INT_VEC3,       "uvec3")
		VCL_ENUM_TO_TYPE(UnsignedInt4,         GL_UNSIGNED_INT_VEC4,       "uvec4")
		VCL_ENUM_TO_TYPE(Bool,                 GL_BOOL,                    "bool")
		VCL_ENUM_TO_TYPE(Bool2,                GL_BOOL_VEC2,               "bvec2")
		VCL_ENUM_TO_TYPE(Bool3,                GL_BOOL_VEC3,               "bvec3")
		VCL_ENUM_TO_TYPE(Bool4,                GL_BOOL_VEC4,               "bvec4")
		VCL_ENUM_TO_TYPE(FloatMatrix2,         GL_FLOAT_MAT2,              "mat2")
		VCL_ENUM_TO_TYPE(FloatMatrix2x3,       GL_FLOAT_MAT2x3,            "mat2x3")
		VCL_ENUM_TO_TYPE(FloatMatrix2x4,       GL_FLOAT_MAT2x4,            "mat2x4")
		VCL_ENUM_TO_TYPE(FloatMatrix3,         GL_FLOAT_MAT3,              "mat3")
		VCL_ENUM_TO_TYPE(FloatMatrix3x2,       GL_FLOAT_MAT3x2,            "mat3x2")
		VCL_ENUM_TO_TYPE(FloatMatrix3x4,       GL_FLOAT_MAT3x4,            "mat3x4")
		VCL_ENUM_TO_TYPE(FloatMatrix4,         GL_FLOAT_MAT4,              "mat4")
		VCL_ENUM_TO_TYPE(FloatMatrix4x2,       GL_FLOAT_MAT4x2,            "mat4x2")
		VCL_ENUM_TO_TYPE(FloatMatrix4x3,       GL_FLOAT_MAT4x3,            "mat4x3")
		VCL_ENUM_TO_TYPE(DoubleMatrix2,        GL_DOUBLE_MAT2,             "dmat2")
		VCL_ENUM_TO_TYPE(DoubleMatrix2x3,      GL_DOUBLE_MAT2x3,           "dmat2x3")
		VCL_ENUM_TO_TYPE(DoubleMatrix2x4,      GL_DOUBLE_MAT2x4,           "dmat2x4")
		VCL_ENUM_TO_TYPE(DoubleMatrix3,        GL_DOUBLE_MAT3,             "dmat3")
		VCL_ENUM_TO_TYPE(DoubleMatrix3x2,      GL_DOUBLE_MAT3x2,           "dmat3x2")
		VCL_ENUM_TO_TYPE(DoubleMatrix3x4,      GL_DOUBLE_MAT3x4,           "dmat3x4")
		VCL_ENUM_TO_TYPE(DoubleMatrix4,        GL_DOUBLE_MAT4,             "dmat4")
		VCL_ENUM_TO_TYPE(DoubleMatrix4x2,      GL_DOUBLE_MAT4x2,           "dmat4x2")
		VCL_ENUM_TO_TYPE(DoubleMatrix4x3,      GL_DOUBLE_MAT4x3,           "dmat4x3")

		VCL_ENUM_TO_TYPE(Sampler1D,                       GL_SAMPLER_1D,                          "sampler1D")
		VCL_ENUM_TO_TYPE(Sampler1DArray,                  GL_SAMPLER_1D_ARRAY,                    "sampler1DArray")
		VCL_ENUM_TO_TYPE(Sampler1DArrayShadow,            GL_SAMPLER_1D_ARRAY_SHADOW,             "sampler1DArrayShadow")
		VCL_ENUM_TO_TYPE(Sampler1DShadow,                 GL_SAMPLER_1D_SHADOW,                   "sampler1DShadow")
		VCL_ENUM_TO_TYPE(Sampler2D,                       GL_SAMPLER_2D,                          "sampler2D")
		VCL_ENUM_TO_TYPE(Sampler2DArray,                  GL_SAMPLER_2D_ARRAY,                    "sampler2DArray")
		VCL_ENUM_TO_TYPE(Sampler2DArrayShadow,            GL_SAMPLER_2D_ARRAY_SHADOW,             "sampler2DArrayShadow")
		VCL_ENUM_TO_TYPE(Sampler2DMultisample,            GL_SAMPLER_2D_MULTISAMPLE,              "sampler2DMultisample")
		VCL_ENUM_TO_TYPE(Sampler2DRect,                   GL_SAMPLER_2D_RECT,                     "sampler2DRect")
		VCL_ENUM_TO_TYPE(Sampler2DRectShadow,             GL_SAMPLER_2D_RECT_SHADOW,              "sampler2DRectShadow")
		VCL_ENUM_TO_TYPE(Sampler2DShadow,                 GL_SAMPLER_2D_SHADOW,                   "sampler2DShadow")
		VCL_ENUM_TO_TYPE(Sampler3D,                       GL_SAMPLER_3D,                          "sampler3D")
		VCL_ENUM_TO_TYPE(SamplerBuffer,                   GL_SAMPLER_BUFFER,                      "samplerBuffer")
		VCL_ENUM_TO_TYPE(SamplerCube,                     GL_SAMPLER_CUBE,                        "samplerCube")
		VCL_ENUM_TO_TYPE(SamplerCubeArray,                GL_SAMPLER_CUBE_MAP_ARRAY,              "samplerCubeArray")
		VCL_ENUM_TO_TYPE(SamplerCubeArrayShadow,          GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW,       "samplerCubeArrayShadow")
		VCL_ENUM_TO_TYPE(SamplerCubeShadow,               GL_SAMPLER_CUBE_SHADOW,                 "samplerCubeShadow")
		VCL_ENUM_TO_TYPE(IntSampler1D,                    GL_INT_SAMPLER_1D,                      "isampler1D")
		VCL_ENUM_TO_TYPE(IntSampler1DArray,               GL_INT_SAMPLER_1D_ARRAY,                "isampler1DArray")
		VCL_ENUM_TO_TYPE(IntSampler2D,                    GL_INT_SAMPLER_2D,                      "isampler2D")
		VCL_ENUM_TO_TYPE(IntSampler2DArray,               GL_INT_SAMPLER_2D_ARRAY,                "isampler2DArray")
		VCL_ENUM_TO_TYPE(IntSampler2DMultisample,         GL_INT_SAMPLER_2D_MULTISAMPLE,          "isampler2DMultisample")
		VCL_ENUM_TO_TYPE(IntSampler2DRect,                GL_INT_SAMPLER_2D_RECT,                 "isampler2DRect")
		VCL_ENUM_TO_TYPE(IntSampler3D,                    GL_INT_SAMPLER_3D,                      "isampler3D")
		VCL_ENUM_TO_TYPE(IntSamplerBuffer,                GL_INT_SAMPLER_BUFFER,                  "isamplerBuffer")
		VCL_ENUM_TO_TYPE(IntSamplerCube,                  GL_INT_SAMPLER_CUBE,                    "isamplerCube")
		VCL_ENUM_TO_TYPE(IntSamplerCubeArray,             GL_INT_SAMPLER_CUBE_MAP_ARRAY,          "isamplerCubeArray")
		VCL_ENUM_TO_TYPE(UnsignedIntSampler1D,            GL_UNSIGNED_INT_SAMPLER_1D,             "usampler1D")
		VCL_ENUM_TO_TYPE(UnsignedIntSampler1DArray,       GL_UNSIGNED_INT_SAMPLER_1D_ARRAY,       "usampler1DArray")
		VCL_ENUM_TO_TYPE(UnsignedIntSampler2D,            GL_UNSIGNED_INT_SAMPLER_2D,             "usampler2D")
		VCL_ENUM_TO_TYPE(UnsignedIntSampler2DArray,       GL_UNSIGNED_INT_SAMPLER_2D_ARRAY,       "usampler2DArray")
		VCL_ENUM_TO_TYPE(UnsignedIntSampler2DMultisample, GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE, "usampler2DMultisample")
		VCL_ENUM_TO_TYPE(UnsignedIntSampler2DRect,        GL_UNSIGNED_INT_SAMPLER_2D_RECT,        "usampler2DRect")
		VCL_ENUM_TO_TYPE(UnsignedIntSampler3D,            GL_UNSIGNED_INT_SAMPLER_3D,             "usampler3D")
		VCL_ENUM_TO_TYPE(UnsignedIntSamplerBuffer,        GL_UNSIGNED_INT_SAMPLER_BUFFER,         "usamplerBuffer")
		VCL_ENUM_TO_TYPE(UnsignedIntSamplerCube,          GL_UNSIGNED_INT_SAMPLER_CUBE,           "usamplerCube")
		VCL_ENUM_TO_TYPE(UnsignedIntSamplerCubeArray,     GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY, "usamplerCubeArray")
		
		VCL_ENUM_TO_TYPE(Image1D,                   GL_IMAGE_1D,                                "image1D")
		VCL_ENUM_TO_TYPE(Image2D,                   GL_IMAGE_2D,                                "image2D")
		VCL_ENUM_TO_TYPE(Image3D,                   GL_IMAGE_3D,                                "image3D")
		VCL_ENUM_TO_TYPE(ImageCube,                 GL_IMAGE_CUBE,                              "imageCube")
		VCL_ENUM_TO_TYPE(Image2DRect,               GL_IMAGE_2D_RECT,                           "image2DRect")
		VCL_ENUM_TO_TYPE(Image1DArray,              GL_IMAGE_1D_ARRAY,                          "image1DArray")
		VCL_ENUM_TO_TYPE(Image2DArray,              GL_IMAGE_2D_ARRAY,                          "image2DArray")
		VCL_ENUM_TO_TYPE(ImageCubeArray,            GL_IMAGE_CUBE_MAP_ARRAY,                    "image2DCubeArray")
		VCL_ENUM_TO_TYPE(ImageBuffer,               GL_IMAGE_BUFFER,                            "imageBuffer")
		VCL_ENUM_TO_TYPE(Image2DMS,                 GL_IMAGE_2D_MULTISAMPLE,                    "image2DMS")
		VCL_ENUM_TO_TYPE(Image2DMSArray,            GL_IMAGE_2D_MULTISAMPLE_ARRAY,              "image2DMSArray")
		VCL_ENUM_TO_TYPE(IntImage1D,                GL_INT_IMAGE_1D,                            "iimage1D")
		VCL_ENUM_TO_TYPE(IntImage2D,                GL_INT_IMAGE_2D,                            "iimage2D")
		VCL_ENUM_TO_TYPE(IntImage3D,                GL_INT_IMAGE_3D,                            "iimage3D")
		VCL_ENUM_TO_TYPE(IntImageCube,              GL_INT_IMAGE_CUBE,                          "iimageCube")
		VCL_ENUM_TO_TYPE(IntImage2DRect,            GL_INT_IMAGE_2D_RECT,                       "iimage2DRect")
		VCL_ENUM_TO_TYPE(IntImage1DArray,           GL_INT_IMAGE_1D_ARRAY,                      "iimage1DArray")
		VCL_ENUM_TO_TYPE(IntImage2DArray,           GL_INT_IMAGE_2D_ARRAY,                      "iimage2DArray")
		VCL_ENUM_TO_TYPE(IntImageCubeArray,         GL_INT_IMAGE_CUBE_MAP_ARRAY,                "iimage2DCubeArray")
		VCL_ENUM_TO_TYPE(IntImageBuffer,            GL_INT_IMAGE_BUFFER,                        "iimageBuffer")
		VCL_ENUM_TO_TYPE(IntImage2DMS,              GL_INT_IMAGE_2D_MULTISAMPLE,                "iimage2DMS")
		VCL_ENUM_TO_TYPE(IntImage2DMSArray,         GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY,          "iimage2DMSArray")
		VCL_ENUM_TO_TYPE(UnsignedIntImage1D,        GL_UNSIGNED_INT_IMAGE_1D,                   "uimage1D")
		VCL_ENUM_TO_TYPE(UnsignedIntImage2D,        GL_UNSIGNED_INT_IMAGE_2D,                   "uimage2D")
		VCL_ENUM_TO_TYPE(UnsignedIntImage3D,        GL_UNSIGNED_INT_IMAGE_3D,                   "uimage3D")
		VCL_ENUM_TO_TYPE(UnsignedIntImageCube,      GL_UNSIGNED_INT_IMAGE_CUBE,                 "uimageCube")
		VCL_ENUM_TO_TYPE(UnsignedIntImage2DRect,    GL_UNSIGNED_INT_IMAGE_2D_RECT,              "uimage2DRect")
		VCL_ENUM_TO_TYPE(UnsignedIntImage1DArray,   GL_UNSIGNED_INT_IMAGE_1D_ARRAY,             "uimage1DArray")
		VCL_ENUM_TO_TYPE(UnsignedIntImage2DArray,   GL_UNSIGNED_INT_IMAGE_2D_ARRAY,             "uimage2DArray")
		VCL_ENUM_TO_TYPE(UnsignedIntImageCubeArray, GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY,       "uimage2DCubeArray")
		VCL_ENUM_TO_TYPE(UnsignedIntImageBuffer,    GL_UNSIGNED_INT_IMAGE_BUFFER,               "uimageBuffer")
		VCL_ENUM_TO_TYPE(UnsignedIntImage2DMS,      GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE,       "uimage2DMS")
		VCL_ENUM_TO_TYPE(UnsignedIntImage2DMSArray, GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY, "uimage2DMSArray")

		default: { VclDebugError("Enumeration value is valid."); }
		}

		return ProgramResourceType::Invalid;
	}

	const char* ProgramResources::name(GLenum type)
	{
		switch (type)
		{
		VCL_ENUM_TO_NAME(Float,                GL_FLOAT,                   "float")
		VCL_ENUM_TO_NAME(Float2,               GL_FLOAT_VEC2,              "vec2")
		VCL_ENUM_TO_NAME(Float3,               GL_FLOAT_VEC3,              "vec3")
		VCL_ENUM_TO_NAME(Float4,               GL_FLOAT_VEC4,              "vec4")
		VCL_ENUM_TO_NAME(Double,               GL_DOUBLE,                  "double")
		VCL_ENUM_TO_NAME(Double2,              GL_DOUBLE_VEC2,             "dvec2")
		VCL_ENUM_TO_NAME(Double3,              GL_DOUBLE_VEC3,             "dvec3")
		VCL_ENUM_TO_NAME(Double4,              GL_DOUBLE_VEC4,             "dvec4")
		VCL_ENUM_TO_NAME(Int,                  GL_INT,                     "int")
		VCL_ENUM_TO_NAME(Int2,                 GL_INT_VEC2,                "ivec2")
		VCL_ENUM_TO_NAME(Int3,                 GL_INT_VEC3,                "ivec3")
		VCL_ENUM_TO_NAME(Int4,                 GL_INT_VEC4,                "ivec4")
		VCL_ENUM_TO_NAME(UnsignedInt,          GL_UNSIGNED_INT,            "uint")
		VCL_ENUM_TO_NAME(UnsignedInt2,         GL_UNSIGNED_INT_VEC2,       "uvec2")
		VCL_ENUM_TO_NAME(UnsignedInt3,         GL_UNSIGNED_INT_VEC3,       "uvec3")
		VCL_ENUM_TO_NAME(UnsignedInt4,         GL_UNSIGNED_INT_VEC4,       "uvec4")
		VCL_ENUM_TO_NAME(Bool,                 GL_BOOL,                    "bool")
		VCL_ENUM_TO_NAME(Bool2,                GL_BOOL_VEC2,               "bvec2")
		VCL_ENUM_TO_NAME(Bool3,                GL_BOOL_VEC3,               "bvec3")
		VCL_ENUM_TO_NAME(Bool4,                GL_BOOL_VEC4,               "bvec4")
		VCL_ENUM_TO_NAME(FloatMatrix2,         GL_FLOAT_MAT2,              "mat2")
		VCL_ENUM_TO_NAME(FloatMatrix2x3,       GL_FLOAT_MAT2x3,            "mat2x3")
		VCL_ENUM_TO_NAME(FloatMatrix2x4,       GL_FLOAT_MAT2x4,            "mat2x4")
		VCL_ENUM_TO_NAME(FloatMatrix3,         GL_FLOAT_MAT3,              "mat3")
		VCL_ENUM_TO_NAME(FloatMatrix3x2,       GL_FLOAT_MAT3x2,            "mat3x2")
		VCL_ENUM_TO_NAME(FloatMatrix3x4,       GL_FLOAT_MAT3x4,            "mat3x4")
		VCL_ENUM_TO_NAME(FloatMatrix4,         GL_FLOAT_MAT4,              "mat4")
		VCL_ENUM_TO_NAME(FloatMatrix4x2,       GL_FLOAT_MAT4x2,            "mat4x2")
		VCL_ENUM_TO_NAME(FloatMatrix4x3,       GL_FLOAT_MAT4x3,            "mat4x3")
		VCL_ENUM_TO_NAME(DoubleMatrix2,        GL_DOUBLE_MAT2,             "dmat2")
		VCL_ENUM_TO_NAME(DoubleMatrix2x3,      GL_DOUBLE_MAT2x3,           "dmat2x3")
		VCL_ENUM_TO_NAME(DoubleMatrix2x4,      GL_DOUBLE_MAT2x4,           "dmat2x4")
		VCL_ENUM_TO_NAME(DoubleMatrix3,        GL_DOUBLE_MAT3,             "dmat3")
		VCL_ENUM_TO_NAME(DoubleMatrix3x2,      GL_DOUBLE_MAT3x2,           "dmat3x2")
		VCL_ENUM_TO_NAME(DoubleMatrix3x4,      GL_DOUBLE_MAT3x4,           "dmat3x4")
		VCL_ENUM_TO_NAME(DoubleMatrix4,        GL_DOUBLE_MAT4,             "dmat4")
		VCL_ENUM_TO_NAME(DoubleMatrix4x2,      GL_DOUBLE_MAT4x2,           "dmat4x2")
		VCL_ENUM_TO_NAME(DoubleMatrix4x3,      GL_DOUBLE_MAT4x3,           "dmat4x3")

		VCL_ENUM_TO_NAME(Sampler1D,                       GL_SAMPLER_1D,                          "sampler1D")
		VCL_ENUM_TO_NAME(Sampler1DArray,                  GL_SAMPLER_1D_ARRAY,                    "sampler1DArray")
		VCL_ENUM_TO_NAME(Sampler1DArrayShadow,            GL_SAMPLER_1D_ARRAY_SHADOW,             "sampler1DArrayShadow")
		VCL_ENUM_TO_NAME(Sampler1DShadow,                 GL_SAMPLER_1D_SHADOW,                   "sampler1DShadow")
		VCL_ENUM_TO_NAME(Sampler2D,                       GL_SAMPLER_2D,                          "sampler2D")
		VCL_ENUM_TO_NAME(Sampler2DArray,                  GL_SAMPLER_2D_ARRAY,                    "sampler2DArray")
		VCL_ENUM_TO_NAME(Sampler2DArrayShadow,            GL_SAMPLER_2D_ARRAY_SHADOW,             "sampler2DArrayShadow")
		VCL_ENUM_TO_NAME(Sampler2DMultisample,            GL_SAMPLER_2D_MULTISAMPLE,              "sampler2DMultisample")
		VCL_ENUM_TO_NAME(Sampler2DRect,                   GL_SAMPLER_2D_RECT,                     "sampler2DRect")
		VCL_ENUM_TO_NAME(Sampler2DRectShadow,             GL_SAMPLER_2D_RECT_SHADOW,              "sampler2DRectShadow")
		VCL_ENUM_TO_NAME(Sampler2DShadow,                 GL_SAMPLER_2D_SHADOW,                   "sampler2DShadow")
		VCL_ENUM_TO_NAME(Sampler3D,                       GL_SAMPLER_3D,                          "sampler3D")
		VCL_ENUM_TO_NAME(SamplerBuffer,                   GL_SAMPLER_BUFFER,                      "samplerBuffer")
		VCL_ENUM_TO_NAME(SamplerCube,                     GL_SAMPLER_CUBE,                        "samplerCube")
		VCL_ENUM_TO_NAME(SamplerCubeArray,                GL_SAMPLER_CUBE_MAP_ARRAY,              "samplerCubeArray")
		VCL_ENUM_TO_NAME(SamplerCubeArrayShadow,          GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW,       "samplerCubeArrayShadow")
		VCL_ENUM_TO_NAME(SamplerCubeShadow,               GL_SAMPLER_CUBE_SHADOW,                 "samplerCubeShadow")
		VCL_ENUM_TO_NAME(IntSampler1D,                    GL_INT_SAMPLER_1D,                      "isampler1D")
		VCL_ENUM_TO_NAME(IntSampler1DArray,               GL_INT_SAMPLER_1D_ARRAY,                "isampler1DArray")
		VCL_ENUM_TO_NAME(IntSampler2D,                    GL_INT_SAMPLER_2D,                      "isampler2D")
		VCL_ENUM_TO_NAME(IntSampler2DArray,               GL_INT_SAMPLER_2D_ARRAY,                "isampler2DArray")
		VCL_ENUM_TO_NAME(IntSampler2DMultisample,         GL_INT_SAMPLER_2D_MULTISAMPLE,          "isampler2DMultisample")
		VCL_ENUM_TO_NAME(IntSampler2DRect,                GL_INT_SAMPLER_2D_RECT,                 "isampler2DRect")
		VCL_ENUM_TO_NAME(IntSampler3D,                    GL_INT_SAMPLER_3D,                      "isampler3D")
		VCL_ENUM_TO_NAME(IntSamplerBuffer,                GL_INT_SAMPLER_BUFFER,                  "isamplerBuffer")
		VCL_ENUM_TO_NAME(IntSamplerCube,                  GL_INT_SAMPLER_CUBE,                    "isamplerCube")
		VCL_ENUM_TO_NAME(IntSamplerCubeArray,             GL_INT_SAMPLER_CUBE_MAP_ARRAY,          "isamplerCubeArray")
		VCL_ENUM_TO_NAME(UnsignedIntSampler1D,            GL_UNSIGNED_INT_SAMPLER_1D,             "usampler1D")
		VCL_ENUM_TO_NAME(UnsignedIntSampler1DArray,       GL_UNSIGNED_INT_SAMPLER_1D_ARRAY,       "usampler1DArray")
		VCL_ENUM_TO_NAME(UnsignedIntSampler2D,            GL_UNSIGNED_INT_SAMPLER_2D,             "usampler2D")
		VCL_ENUM_TO_NAME(UnsignedIntSampler2DArray,       GL_UNSIGNED_INT_SAMPLER_2D_ARRAY,       "usampler2DArray")
		VCL_ENUM_TO_NAME(UnsignedIntSampler2DMultisample, GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE, "usampler2DMultisample")
		VCL_ENUM_TO_NAME(UnsignedIntSampler2DRect,        GL_UNSIGNED_INT_SAMPLER_2D_RECT,        "usampler2DRect")
		VCL_ENUM_TO_NAME(UnsignedIntSampler3D,            GL_UNSIGNED_INT_SAMPLER_3D,             "usampler3D")
		VCL_ENUM_TO_NAME(UnsignedIntSamplerBuffer,        GL_UNSIGNED_INT_SAMPLER_BUFFER,         "usamplerBuffer")
		VCL_ENUM_TO_NAME(UnsignedIntSamplerCube,          GL_UNSIGNED_INT_SAMPLER_CUBE,           "usamplerCube")
		VCL_ENUM_TO_NAME(UnsignedIntSamplerCubeArray,     GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY, "usamplerCubeArray")
		
		VCL_ENUM_TO_NAME(Image1D,                   GL_IMAGE_1D,                                "image1D")
		VCL_ENUM_TO_NAME(Image2D,                   GL_IMAGE_2D,                                "image2D")
		VCL_ENUM_TO_NAME(Image3D,                   GL_IMAGE_3D,                                "image3D")
		VCL_ENUM_TO_NAME(ImageCube,                 GL_IMAGE_CUBE,                              "imageCube")
		VCL_ENUM_TO_NAME(Image2DRect,               GL_IMAGE_2D_RECT,                           "image2DRect")
		VCL_ENUM_TO_NAME(Image1DArray,              GL_IMAGE_1D_ARRAY,                          "image1DArray")
		VCL_ENUM_TO_NAME(Image2DArray,              GL_IMAGE_2D_ARRAY,                          "image2DArray")
		VCL_ENUM_TO_NAME(ImageCubeArray,            GL_IMAGE_CUBE_MAP_ARRAY,                    "image2DCubeArray")
		VCL_ENUM_TO_NAME(ImageBuffer,               GL_IMAGE_BUFFER,                            "imageBuffer")
		VCL_ENUM_TO_NAME(Image2DMS,                 GL_IMAGE_2D_MULTISAMPLE,                    "image2DMS")
		VCL_ENUM_TO_NAME(Image2DMSArray,            GL_IMAGE_2D_MULTISAMPLE_ARRAY,              "image2DMSArray")
		VCL_ENUM_TO_NAME(IntImage1D,                GL_INT_IMAGE_1D,                            "iimage1D")
		VCL_ENUM_TO_NAME(IntImage2D,                GL_INT_IMAGE_2D,                            "iimage2D")
		VCL_ENUM_TO_NAME(IntImage3D,                GL_INT_IMAGE_3D,                            "iimage3D")
		VCL_ENUM_TO_NAME(IntImageCube,              GL_INT_IMAGE_CUBE,                          "iimageCube")
		VCL_ENUM_TO_NAME(IntImage2DRect,            GL_INT_IMAGE_2D_RECT,                       "iimage2DRect")
		VCL_ENUM_TO_NAME(IntImage1DArray,           GL_INT_IMAGE_1D_ARRAY,                      "iimage1DArray")
		VCL_ENUM_TO_NAME(IntImage2DArray,           GL_INT_IMAGE_2D_ARRAY,                      "iimage2DArray")
		VCL_ENUM_TO_NAME(IntImageCubeArray,         GL_INT_IMAGE_CUBE_MAP_ARRAY,                "iimage2DCubeArray")
		VCL_ENUM_TO_NAME(IntImageBuffer,            GL_INT_IMAGE_BUFFER,                        "iimageBuffer")
		VCL_ENUM_TO_NAME(IntImage2DMS,              GL_INT_IMAGE_2D_MULTISAMPLE,                "iimage2DMS")
		VCL_ENUM_TO_NAME(IntImage2DMSArray,         GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY,          "iimage2DMSArray")
		VCL_ENUM_TO_NAME(UnsignedIntImage1D,        GL_UNSIGNED_INT_IMAGE_1D,                   "uimage1D")
		VCL_ENUM_TO_NAME(UnsignedIntImage2D,        GL_UNSIGNED_INT_IMAGE_2D,                   "uimage2D")
		VCL_ENUM_TO_NAME(UnsignedIntImage3D,        GL_UNSIGNED_INT_IMAGE_3D,                   "uimage3D")
		VCL_ENUM_TO_NAME(UnsignedIntImageCube,      GL_UNSIGNED_INT_IMAGE_CUBE,                 "uimageCube")
		VCL_ENUM_TO_NAME(UnsignedIntImage2DRect,    GL_UNSIGNED_INT_IMAGE_2D_RECT,              "uimage2DRect")
		VCL_ENUM_TO_NAME(UnsignedIntImage1DArray,   GL_UNSIGNED_INT_IMAGE_1D_ARRAY,             "uimage1DArray")
		VCL_ENUM_TO_NAME(UnsignedIntImage2DArray,   GL_UNSIGNED_INT_IMAGE_2D_ARRAY,             "uimage2DArray")
		VCL_ENUM_TO_NAME(UnsignedIntImageCubeArray, GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY,       "uimage2DCubeArray")
		VCL_ENUM_TO_NAME(UnsignedIntImageBuffer,    GL_UNSIGNED_INT_IMAGE_BUFFER,               "uimageBuffer")
		VCL_ENUM_TO_NAME(UnsignedIntImage2DMS,      GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE,       "uimage2DMS")
		VCL_ENUM_TO_NAME(UnsignedIntImage2DMSArray, GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY, "uimage2DMSArray")

		default: { VclDebugError("Enumeration value is valid."); }
		}

		return "";
	}

	ShaderProgram::ShaderProgram(const ShaderProgramDescription& desc)
	{
		VclRequire
		(
			implies
			(
				desc.ComputeShader, 
				!(desc.VertexShader || desc.TessControlShader || desc.TessEvalShader || desc.GeometryShader || desc.FragmentShader)
			),
			"Compute shader is not combined with graphics shaders."
		);
		VclRequire(implies(desc.ComputeShader, glewIsExtensionSupported("GL_ARB_compute_shader")), "Compute shaders are supported.");
		VclRequire(implies(!desc.ComputeShader, desc.VertexShader && desc.FragmentShader), "VertexShader and Fragment shader are set.");

		// Create a program for the shader
		_glId = glCreateProgram();

		// Attach shaders to the program
		if (desc.ComputeShader)
		{
			glAttachShader(_glId, desc.ComputeShader->id());
		}
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

		bool linked = checkLinkState();

		// Link the program to the input layout
		if (linked && !desc.ComputeShader)
			linkAttributes(desc.InputLayout);

		// Detach shaders for deferred deletion
		if (desc.ComputeShader)
		{
			glDetachShader(_glId, desc.ComputeShader->id());
		}
		else
		{
			if (desc.VertexShader)
				glDetachShader(_glId, desc.VertexShader->id());

			if (desc.TessControlShader)
				glDetachShader(_glId, desc.TessControlShader->id());

			if (desc.TessEvalShader)
				glDetachShader(_glId, desc.TessEvalShader->id());

			if (desc.GeometryShader)
				glDetachShader(_glId, desc.GeometryShader->id());

			if (desc.FragmentShader)
				glDetachShader(_glId, desc.FragmentShader->id());
		}

		// Collect the uniforms of this program
		if (linked)
			_resources = std::make_unique<ProgramResources>(id());

		VclEnsure(id() > 0, "Shader program is created");
	}

	void ShaderProgram::bind()
	{
		VclRequire(id() > 0, "Shader program is created");

		glUseProgram(id());
	}


	void ShaderProgram::linkAttributes(const InputLayoutDescription& layout)
	{
		// If no layout is set, nothing needs to be done
		if (layout.size() == 0)
			return;

		// Match the attributes against the interface provided in the program description
		// The interface may contain more data than the shader can consume.
		// The interface must provide at least the used by the shader.
		ProgramAttributes attribs{ _glId };
		for (const auto& attrib : attribs)
		{
			const auto& name = attrib.Name;

			auto e = std::find_if(std::begin(layout), std::end(layout), [name](const InputLayoutElement& e)
			{
				return e.Name == name;
			});

			// Bind the layout entry to the shader input variable
			if (e != std::end(layout))
			{
				size_t idx = e - std::begin(layout);
				int loc = layout.location(idx);

				glBindAttribLocation(_glId, loc, name.c_str());
			}
			else if (name.find("gl_") != 0)
			{
				// Append to error output
				VclDebugError("Attribute could not be matched.");
			}
		}

		// Link the fragment output against the layout
		//if (mFragmentLayout)
		//{
		//	for (unsigned int i = 0; i < mFragmentLayout->count(); i++)
		//	{
		//		glBindFragDataLocation(mShaderProgramObject, i, mFragmentLayout->operator[](i).Name.c_str());
		//	}
		//}

		// Relink the program to enable the changed binding configuration
		glLinkProgram(id());

		VclAssertBlock
		{
			// Check the input layout against the description
			int idx = 0;
			for (const auto& attrib : layout)
			{
				const auto& name = attrib.Name;
				int loc = layout.location(idx++);

				VclCheckEx(implies(glGetAttribLocation(_glId, name.c_str()) >= 0, glGetAttribLocation(_glId, name.c_str()) == loc), "Input layout element is bound correctly", fmt::format("Attribute: {}; GL location: {}; input location: {}", name, glGetAttribLocation(_glId, name.c_str()), loc));
			}

			// Check the fragment output against the layout
			//if (mFragmentLayout)
			//{
			//	for (unsigned int i = 0; i < mFragmentLayout->count(); i++)
			//	{
			//		VclCheck(glGetFragDataLocation(mShaderProgramObject, mFragmentLayout->operator[](i).Name.c_str()) == (int) i, "Fragment output is bound correctly.");
			//	}
			//}
		}
	}

	bool ShaderProgram::checkLinkState() const
	{
		GLint linked = GL_FALSE;
		glGetProgramiv(id(), GL_LINK_STATUS, &linked);
		return linked == GL_TRUE;
	}

	bool ShaderProgram::validate() const
	{
		glValidateProgram(id());

		GLint valid = GL_FALSE;
		glGetProgramiv(id(), GL_VALIDATE_STATUS, &valid);
		return valid == GL_TRUE;
	}

	std::string ShaderProgram::readInfoLog() const
	{
		if (_glId == 0)
			return{};

		int info_log_length = 0;
		int chars_written = 0;
		glGetProgramiv(_glId, GL_INFO_LOG_LENGTH, &info_log_length);
		if (info_log_length > 1)
		{
			std::string info_log(info_log_length, '\0');
			glGetProgramInfoLog(_glId, info_log_length, &chars_written, const_cast<char*>(info_log.data()));
			return info_log;
		}
		return{};
	}

	UniformHandle ShaderProgram::uniform(const char* name) const
	{
		const auto& uniforms = _resources->uniforms();
		auto handle = std::find_if(std::begin(uniforms), std::end(uniforms), [name](const UniformData& data)
		{
			return strcmp(data.Name.c_str(), name) == 0;
		});
		if (handle != std::end(uniforms))
		{
			return{ id(), (short) handle->Location, (short) handle->ResourceLocation };
		}
		else
		{
			return{ id(), -1, -1 };
		}
	}

	void ShaderProgram::setUniform(const UniformHandle& handle, float value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform1f(id(), handle.Location, value);
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector2f& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform2f(id(), handle.Location, value.x(), value.y());
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector3f& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform3f(id(), handle.Location, value.x(), value.y(), value.z());
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector4f& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform4f(id(), handle.Location, value.x(), value.y(), value.z(), value.w());
	}

	void ShaderProgram::setUniform(const UniformHandle& handle, int value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform1i(id(), handle.Location, value);
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector2i& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform2i(id(), handle.Location, value.x(), value.y());
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector3i& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform3i(id(), handle.Location, value.x(), value.y(), value.z());
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector4i& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform4i(id(), handle.Location, value.x(), value.y(), value.z(), value.w());
	}

	void ShaderProgram::setUniform(const UniformHandle& handle, unsigned int value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform1ui(id(), handle.Location, value);
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector2ui& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform2ui(id(), handle.Location, value.x(), value.y());
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector3ui& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform3ui(id(), handle.Location, value.x(), value.y(), value.z());
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Vector4ui& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniform4ui(id(), handle.Location, value.x(), value.y(), value.z(), value.w());
	}
	void ShaderProgram::setUniform(const UniformHandle& handle, const Eigen::Matrix4f& value)
	{
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		glProgramUniformMatrix4fv(id(), handle.Location, 1, GL_FALSE, value.data());
	}

	void ShaderProgram::setTexture(const UniformHandle& handle, const Runtime::Texture* tex, const Runtime::Sampler* sampler)
	{
		VclRequire(dynamic_cast<const OpenGL::Texture*>(tex), "'img' is from the OpenGL backend");
		VclRequire(implies(sampler, dynamic_cast<const OpenGL::Sampler*>(sampler)), "'sampler' is from the OpenGL backend");
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		if (handle.Location >= 0)
		{
			auto res = static_cast<const OpenGL::Texture*>(tex);
			glBindTextureUnit(handle.ResourceLocation, res->id());
			if (sampler)
				glBindSampler(handle.ResourceLocation, static_cast<const OpenGL::Sampler*>(sampler)->id());
		}
	}

	void ShaderProgram::setImage(const UniformHandle& handle, const Runtime::Texture* img, bool read, bool write)
	{
		VclRequire(dynamic_cast<const OpenGL::Texture*>(img), "'img' is from the OpenGL backend");
		VclRequire(id() == handle.Program, "Handle belongs to this program");

		if (handle.Location >= 0)
		{
			GLenum access = GL_NONE;
			if (read && !write)
				access |= GL_READ_ONLY;
			if (write && !read)
				access |= GL_WRITE_ONLY;
			if (write && read)
				access |= GL_READ_WRITE;

			auto res = static_cast<const OpenGL::Texture*>(img);
			glBindImageTexture(handle.ResourceLocation, res->id(), 0, false, 0, access, Texture::toSurfaceFormat(img->format()));
		}
	}

	void ShaderProgram::setConstantBuffer(const char* name, const Runtime::Buffer* buf, size_t offset, size_t size)
	{
		VclRequire(dynamic_cast<const OpenGL::Buffer*>(buf), "'buf' is from the OpenGL backend");
		VclRequire(offset + size < buf->sizeInBytes(), "Buffer region is valid.");

		const auto& uniforms = _resources->uniformBlocks();
		auto handle = std::find_if(std::begin(uniforms), std::end(uniforms), [name](const UniformBlockData& data)
		{
			return strcmp(data.Name.c_str(), name) == 0;
		});
		if (handle != std::end(uniforms))
		{
			if (size == 0)
				size = buf->sizeInBytes() - offset;

			auto buffer = static_cast<const OpenGL::Buffer*>(buf);
			glBindBufferRange(GL_UNIFORM_BUFFER, handle->ResourceLocation, buffer->id(), offset, size);

			VclEnsure(static_cast<int>(buffer->id()) == Graphics::OpenGL::GL::getInteger(GL_UNIFORM_BUFFER_BINDING, handle->ResourceLocation), "Buffer is bound.");
		}
	}

	void ShaderProgram::setBuffer(const char* name, const Runtime::Buffer* buf, size_t offset, size_t size)
	{
		VclRequire(_resources, "Resources are defined.");
		VclRequire(dynamic_cast<const OpenGL::Buffer*>(buf), "'buf' is from the OpenGL backend");
		VclRequire(offset + size < buf->sizeInBytes(), "Buffer region is valid.");

		const auto& uniforms = _resources->buffers();
		auto handle = std::find_if(std::begin(uniforms), std::end(uniforms), [name](const BufferBlockData& data)
		{
			return strcmp(data.Name.c_str(), name) == 0;
		});
		if (handle != std::end(uniforms))
		{
			if (size == 0)
				size = buf->sizeInBytes() - offset;

			auto buffer = static_cast<const OpenGL::Buffer*>(buf);
			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, handle->ResourceLocation, buffer->id(), offset, size);

			VclEnsure(static_cast<int>(buffer->id()) == Graphics::OpenGL::GL::getInteger(GL_SHADER_STORAGE_BUFFER_BINDING, handle->ResourceLocation), "Buffer is bound.");
		}
	}

	nonstd::expected<std::unique_ptr<ShaderProgram>, std::string> makeShaderProgram(const ShaderProgramDescription& desc)
	{
		auto prog = std::make_unique<ShaderProgram>(desc);
		if (prog->checkLinkState())
			return std::move(prog);
		else
			return nonstd::make_unexpected(prog->readInfoLog());
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
