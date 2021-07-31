/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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

// VCL configuration
#include <vcl/config/global.h>

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>

// Google test
#include <gtest/gtest.h>

// Tests the shader compilation
const char* MaxBindingsVS =
R"(
#version 440 core

layout(location = 0) in vec2 Position;
layout(location = 0) out PerVertexData
{
	vec3 Colour;
} Out;

layout(binding =  0) uniform U00 { float u00; };

layout(std430, binding =  0) buffer B00 { float b00; };
layout(std430, binding =  1) buffer B01 { float b01; };
layout(std430, binding =  2) buffer B02 { float b02; };
layout(std430, binding =  3) buffer B03 { float b03; };
layout(std430, binding =  4) buffer B04 { float b04; };
layout(std430, binding =  5) buffer B05 { float b05; };
layout(std430, binding =  6) buffer B06 { float b06; };
layout(std430, binding =  7) buffer B07 { float b07; };

void main()
{
	float b = u00 + b00 + b01 + b02 + b03 + b04 + b05 + b06 + b07;
	gl_Position = vec4(Position, b, 1);
}
)";

const char* MaxBindingsFS =
R"(
#version 440 core

layout(location = 0) in PerVertexData
{
	vec3 Colour;
} In;
layout(location = 0) out vec4 Colour;

layout(std430, binding =  8) buffer B10 { float b10; };
layout(std430, binding = 31) buffer B11 { float b11; };
layout(std430, binding = 32) buffer B12 { float b12; };
layout(std430, binding = 33) buffer B13 { float b13; };
layout(std430, binding = 34) buffer B14 { float b14; };
layout(std430, binding = 35) buffer B15 { float b15; };
layout(std430, binding = 36) buffer B16 { float b16; };
layout(std430, binding = 37) buffer B17 { float b17; };

void main()
{	
	float b = b10 + b11 + b12 + b13 + b14 + b15 + b16 + b17;
	Colour = vec4(b);
}
)";

TEST(OpenGL, MaxBindingPoints)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	int max_vert_bindings = 0;
	int max_frag_bindings = 0;
	int max_comb_bindings = 0;
	int max_bindings = 0;
	glGetIntegerv(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, &max_vert_bindings);
	glGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &max_frag_bindings);
	glGetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, &max_comb_bindings);
	glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &max_bindings);

	// Compile the shader stages
	auto vs = Runtime::OpenGL::makeShader(ShaderType::VertexShader, 0, MaxBindingsVS);
	EXPECT_TRUE(vs) << vs.get_unexpected().value();
	auto fs = Runtime::OpenGL::makeShader(ShaderType::FragmentShader, 0, MaxBindingsFS);
	EXPECT_TRUE(fs) << fs.get_unexpected().value();

	// Create the input definition
	InputLayoutDescription in = {
		{
			{ 0, sizeof(Eigen::Vector2f) + sizeof(Eigen::Vector3f), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerObject },
		},
		{
			{ "Position", Vcl::Graphics::SurfaceFormat::R32G32_FLOAT, 0, 0, 0 },
		}
	};

	// Create the program descriptor
	Runtime::OpenGL::ShaderProgramDescription desc;
	desc.InputLayout = in;
	desc.VertexShader = &*vs;
	desc.FragmentShader = &*fs;

	// Create the shader program
	auto prog = Runtime::OpenGL::makeShaderProgram(desc);
	EXPECT_TRUE(prog) << prog.error();

	prog.value()->bind();

	BufferDescription buf_desc = {
		1024,
		{}
	};
	Runtime::OpenGL::Buffer buf(buf_desc);
	std::vector<GLuint> ids(max_bindings, buf.id());
	glBindBuffersBase(GL_SHADER_STORAGE_BUFFER, 0, 8, ids.data());
	glBindBuffersBase(GL_SHADER_STORAGE_BUFFER, 31, 8, ids.data());

	for (int i = 0; i < 8; i++)
	{
		GLint id = 0;
		glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, i, &id);
		EXPECT_EQ(id, buf.id());
	}
	for (int i = 0; i < 8; i++)
	{
		GLint id = 0;
		glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, i + 31, &id);
		EXPECT_EQ(id, buf.id());
	}
}
