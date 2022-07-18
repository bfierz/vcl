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
#include <vcl/config/opengl.h>

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>

// Google test
#include <gtest/gtest.h>

// Additional shaders
#include "quad.vert.spv.h"
#include "quad.frag.spv.h"

// Tests the shader compilation
const char* QuadVS =
	R"(
#version 440 core

layout(location = 0) in vec2 Position;
layout(location = 1) in vec4 Colour;
layout(location = 2) in mat4 Scale;

layout(location = 0) out PerVertexData
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
	Out.Colour = ColorScale*Colour.rgb;
}
)";

const char* QuadFS =
	R"(
#version 440 core

layout(location = 0) in PerVertexData
{
	vec3 Colour;
} In;

layout(location = 0) uniform float alpha = 0.7f;

layout(location = 0) out vec4 Colour;

void main()
{	
	Colour = vec4(In.Colour, alpha);
}
)";

const char* QuadColourAlphaFS =
	R"(
#version 440 core

layout(location = 0) in PerVertexData
{
	vec4 Colour;
} In;

layout(location = 0) uniform float alpha = 0.7f;

layout(location = 0) out vec4 Colour;

void main()
{	
	Colour = In.Colour;
}
)";

const char* SimpleMS =
	R"(
#version 450
 
#extension GL_NV_mesh_shader : require
 
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
 
// Custom vertex output block
layout (location = 0) out PerVertexData
{
  vec4 color;
} v_out[];  // [max_vertices]
 
 
const vec3 vertices[3] = {vec3(-1,-1,0), vec3(0,1,0), vec3(1,-1,0)};
const vec3 colors[3] = {vec3(1.0,0.0,0.0), vec3(0.0,1.0,0.0), vec3(0.0,0.0,1.0)};
 
void main()
{
  // Vertices position
  gl_MeshVerticesNV[0].gl_Position = vec4(vertices[0], 1.0); 
  gl_MeshVerticesNV[1].gl_Position = vec4(vertices[1], 1.0); 
  gl_MeshVerticesNV[2].gl_Position = vec4(vertices[2], 1.0); 
 
  // Vertices color
  v_out[0].color = vec4(colors[0], 1.0);
  v_out[1].color = vec4(colors[1], 1.0);
  v_out[2].color = vec4(colors[2], 1.0);
 
  // Triangle indices
  gl_PrimitiveIndicesNV[0] = 0;
  gl_PrimitiveIndicesNV[1] = 1;
  gl_PrimitiveIndicesNV[2] = 2;
 
  // Number of triangles  
  gl_PrimitiveCountNV = 1;
}
)";

const char* SimpleCS =
	R"(
#version 440 core

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

TEST(OpenGL, CompileShaderError)
{
	using namespace Vcl::Graphics::Runtime;

	// Compile the shader
	auto shader = OpenGL::makeShader(ShaderType::VertexShader, 0, "No Content");
	EXPECT_FALSE(shader);
}

TEST(OpenGL, LinkShaderProgramError)
{
	using namespace Vcl::Graphics::Runtime;

	// Compile the shader stages
	OpenGL::Shader vs(ShaderType::VertexShader, 0, QuadVS);
	OpenGL::Shader fs(ShaderType::FragmentShader, 0, QuadColourAlphaFS);

	// Compile the shader
	OpenGL::ShaderProgramDescription desc;
	desc.VertexShader = &vs;
	desc.FragmentShader = &fs;

	auto prog = OpenGL::makeShaderProgram(desc);
	EXPECT_FALSE(prog);
}

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
	Runtime::OpenGL::Shader vs(ShaderType::VertexShader, 0, QuadVS);
	Runtime::OpenGL::Shader fs(ShaderType::FragmentShader, 0, QuadFS);

	// Create the input definition
	// clang-format off
	InputLayoutDescription in = {
		{
			{ 0, sizeof(Eigen::Vector2f) + sizeof(Eigen::Vector3f), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerObject },
			{ 1, sizeof(Eigen::Vector4f), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerInstance }
		},
		{
			{ "Position",  Vcl::Graphics::SurfaceFormat::R32G32_FLOAT, 0, 0, 0 },
			{ "Colour", Vcl::Graphics::SurfaceFormat::R32G32B32_FLOAT, 0, 0, 8 },
			{ "Scale",  Vcl::Graphics::SurfaceFormat::R32G32B32A32_FLOAT, 0, 1, 0 }
		}
	};
	// clang-format on

	// Create the program descriptor
	Runtime::OpenGL::ShaderProgramDescription desc;
	desc.InputLayout = in;
	desc.VertexShader = &vs;
	desc.FragmentShader = &fs;

	// Create the shader program
	auto prog = Runtime::OpenGL::makeShaderProgram(desc);
	EXPECT_TRUE(prog) << prog.error();
}

TEST(OpenGL, BuildSimpleSpirvGraphicsShaderProgram)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	if (!glewIsExtensionSupported("GL_ARB_gl_spirv"))
	{
		std::cout << "[ SKIPPED  ] SPIR-V not supported" << std::endl;
		return;
	}

	// Compile the shader stages
	Runtime::OpenGL::Shader vs(ShaderType::VertexShader, 0, QuadSpirvVS);
	Runtime::OpenGL::Shader fs(ShaderType::FragmentShader, 0, QuadSpirvFS);

	// Create the input definition
	// clang-format off
	InputLayoutDescription in = {
		{
			{ 0, sizeof(Eigen::Vector2f) + sizeof(Eigen::Vector3f), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerObject },
			{ 1, sizeof(Eigen::Vector4f), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerInstance }
		},
		{
			{ "Position",  Vcl::Graphics::SurfaceFormat::R32G32_FLOAT, 0, 0, 0 },
			{ "Colour", Vcl::Graphics::SurfaceFormat::R32G32B32_FLOAT, 0, 0, 8 },
			{ "Scale",  Vcl::Graphics::SurfaceFormat::R32G32B32A32_FLOAT, 0, 1, 0 }
		}
	};
	// clang-format on

	// Create the program descriptor
	Runtime::OpenGL::ShaderProgramDescription desc;
	desc.InputLayout = in;
	desc.VertexShader = &vs;
	desc.FragmentShader = &fs;

	// Create the shader program
	auto prog = Runtime::OpenGL::makeShaderProgram(desc);
	EXPECT_TRUE(prog) << prog.error();
}

TEST(OpenGL, BuildSimpleMeshShader)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	if (!glewIsExtensionSupported("GL_NV_mesh_shader"))
	{
		std::cout << "[ SKIPPED  ] NVIDIA Mesh Shaders not supported" << std::endl;
		return;
	}
	// Compile the shader
	auto ms = Runtime::OpenGL::makeShader(ShaderType::MeshShader, 0, SimpleMS);
	EXPECT_TRUE(ms) << ms.error();

	// Create the program descriptor
	Runtime::OpenGL::MeshShaderProgramDescription desc;
	desc.MeshShader = &ms.value();

	// Create the shader program
	auto prog = Runtime::OpenGL::makeShaderProgram(desc);
	EXPECT_TRUE(prog) << prog.error();
}

TEST(OpenGL, BuildSimpleComputeShaderProgram)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	// Compile the shader
	auto cs = Runtime::OpenGL::makeShader(ShaderType::ComputeShader, 0, SimpleCS);
	EXPECT_TRUE(cs) << cs.error();

	// Create the program descriptor
	Runtime::OpenGL::ShaderProgramDescription desc;
	desc.ComputeShader = &cs.value();

	// Create the shader program
	auto prog = Runtime::OpenGL::makeShaderProgram(desc);
	EXPECT_TRUE(prog) << prog.error();
}

TEST(OpenGL, RunSimpleComputeShaderProgram)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	// Compile the shader
	Runtime::OpenGL::Shader cs(ShaderType::ComputeShader, 0, SimpleCS);

	// Create the program descriptor
	Runtime::OpenGL::ShaderProgramDescription desc;
	desc.ComputeShader = &cs;

	// Create the shader program
	Runtime::OpenGL::ShaderProgram prog{ desc };

	// Create an output image
	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 1;
	desc2d.Width = 256;
	desc2d.Height = 256;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2D output{ desc2d };

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
