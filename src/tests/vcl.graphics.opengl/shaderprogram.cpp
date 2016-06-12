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
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>

// Google test
#include <gtest/gtest.h>

// Tests the shader compilation

const char* QuadVS =
R"(
#version 430 core

in vec2 Position;
in vec3 Colour;
in mat4 Scale;

out PerVertexData
{
	vec3 Colour;
} Out;

layout(binding = 1) uniform MatrixBlock0
{
  mat4 Modelview;
};

uniform MatrixBlock1
{
  mat4 Projection;
};

layout(std430, binding = 2) buffer Colors
{
  vec3 ColorScale;
};

void main()
{
	gl_Position = Projection*Modelview*Scale*vec4(Position, 0, 1);
	Out.Colour = ColorScale*Colour;
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

const char* SimpleCS =
R"(
#version 430 core

// Kernel output
layout(rgba8) uniform image2D output0;

// Size of the local tile
layout(local_size_x = 16, local_size_y = 16) in;

void main()
{
	ivec2 outPos = ivec2(gl_GlobalInvocationID.xy);

	imageStore(output0, outPos, vec4(float(gl_LocalInvocationID.x + gl_LocalInvocationID.y) / 256.0f, 0, 0, 1));
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

TEST(OpenGL, CompileComputeShader)
{
	using namespace Vcl::Graphics::Runtime;

	// Compile the shader
	OpenGL::Shader cs(ShaderType::ComputeShader, 0, SimpleCS);

	// Verify the result
	GLint compiled = 0;
	glGetShaderiv(cs.id(), GL_COMPILE_STATUS, &compiled);

	EXPECT_TRUE(compiled != 0) << "Shader not compiled.";
}

TEST(OpenGL, BuildSimpleGraphicsShaderProgram)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	// Compile the shader stages
	OpenGL::Shader vs(ShaderType::VertexShader, 0, QuadVS);
	OpenGL::Shader fs(ShaderType::FragmentShader, 0, QuadFS);

	// Create the input definition
	InputLayoutDescription in = 
	{
		{ "Scale", SurfaceFormat::R32G32B32A32_FLOAT, 4, 1, 0, VertexDataClassification::VertexDataPerInstance, 0 },
		{ "Position", SurfaceFormat::R32G32_FLOAT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
		{ "Colour", SurfaceFormat::R32G32B32_FLOAT, 0, 0, 8, VertexDataClassification::VertexDataPerObject, 0 },
	};

	// Create the program descriptor
	OpenGL::ShaderProgramDescription desc;
	desc.InputLayout = in;
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

TEST(OpenGL, BuildSimpleComputeShaderProgram)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	// Compile the shader
	OpenGL::Shader cs(ShaderType::ComputeShader, 0, SimpleCS);

	// Create the program descriptor
	OpenGL::ShaderProgramDescription desc;
	desc.ComputeShader = &cs;

	// Create the shader program
	OpenGL::ShaderProgram prog{ desc };

	// Verify the result
	GLint linked = 0, valid = 0;
	glGetProgramiv(prog.id(), GL_LINK_STATUS, &linked);
	glGetProgramiv(prog.id(), GL_VALIDATE_STATUS, &valid);

	EXPECT_TRUE(linked != 0) << "Shader program not linked.";
	EXPECT_TRUE(valid != 0) << "Shader program not valid.";
}

TEST(OpenGL, RunSimpleComputeShaderProgram)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	// Compile the shader
	OpenGL::Shader cs(ShaderType::ComputeShader, 0, SimpleCS);

	// Create the program descriptor
	OpenGL::ShaderProgramDescription desc;
	desc.ComputeShader = &cs;

	// Create the shader program
	OpenGL::ShaderProgram prog{ desc };

	// Create an output image
	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 1;
	desc2d.Width = 256;
	desc2d.Height = 256;
	desc2d.MipLevels = 1;
	OpenGL::Texture2D output{ desc2d };

	// Bind the program to the pipeline
	prog.bind();

	// Bind the output parameter
	auto h = prog.uniform("output0");
	prog.setImage(h, &output, false, true);

	// Execute the compute shader
	glDispatchCompute(16, 16, 1);

	// Read the generated image
	std::vector<std::array<unsigned char, 4>> pattern(256 * 256, { 254, 254, 254, 254 });
	output.read(pattern.size() * 4, pattern.data());

	// Verify output
	for (int x = 0; x < 16; x++)
	{
		for (int y = 0; y < 16; y++)
		{
			int lx = x % 16;
			int ly = y % 16;

			auto item = pattern[y * 256 + x];
			EXPECT_EQ(item[0], lx + ly) << "Computed pattern is wrong";
			EXPECT_EQ(item[1], 0) << "Computed pattern is wrong";
			EXPECT_EQ(item[2], 0) << "Computed pattern is wrong";
			EXPECT_EQ(item[3], 255) << "Computed pattern is wrong";
		}
	}
}
