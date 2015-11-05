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

// VCL configuration
#include <vcl/config/global.h>

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>

// Google test
#include <gtest/gtest.h>

// Tests the shader compilation

const char* QuadVS =
R"(
#version 400 core

in vec2 Position;
in vec3 Colour;

out PerVertexData
{
	vec3 Colour;
} Out;

void main()
{
	gl_Position = vec4(Position, 0, 1);
	Out.Colour = Colour;
}
)";

const char* QuadFS =
R"(
#version 400 core

in PerVertexData
{
	vec3 Colour;
} In;

uniform float alpha = 0.7f;

out vec4 Colour;

void main()
{	
	Colour = vec4(In.Colour, alpha);
}
)";


TEST(OpenGL, CompileVertexShader)
{
	using namespace Vcl::Graphics::Runtime;

	// Compile the shader
	OpenGL::Shader vs(ShaderType::VertexShader, 0, QuadVS);

	// Verify the result
	GLint compiled = 0;
	glGetShaderiv(vs.id(), GL_COMPILE_STATUS, &compiled);

	EXPECT_TRUE(compiled != 0) << "Shader not compiled.";
}

TEST(OpenGL, BuildSimpleShaderProgram)
{
	using namespace Vcl::Graphics::Runtime;

	// Compile the shader stages
	OpenGL::Shader vs(ShaderType::VertexShader, 0, QuadVS);
	OpenGL::Shader fs(ShaderType::FragmentShader, 0, QuadFS);

	// Create the program descriptor
	OpenGL::ShaderProgramDescription desc;
	desc.VertexShader = &vs;
	desc.FragmentShader = &fs;

	// Create the shader program
	OpenGL::ShaderProgram prog{ desc };

	// Verify the result
	GLint linked = 0, valid = 0;
	glGetProgramiv(prog.id(), GL_LINK_STATUS, &linked);
	glGetProgramiv(prog.id(), GL_VALIDATE_STATUS, &valid);

	EXPECT_TRUE(linked != 0) << "Shader program not linked.";
	EXPECT_TRUE(valid != 0) << "Shader program not valid.";
}
