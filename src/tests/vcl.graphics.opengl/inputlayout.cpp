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
#include <vcl/graphics/opengl/gl.h>
#include <vcl/graphics/runtime/opengl/state/inputlayout.h>

// Google test
#include <gtest/gtest.h>

namespace {
#if defined(VCL_GL_ARB_direct_state_access)
	GLint vertexAttribiv(GLuint vao, GLuint i, GLenum param)
	{
		GLint value = 0;
		glGetVertexArrayIndexediv(vao, i, param, &value);
		return value;
	}
#else
	GLint vertexAttribiv(GLuint vao, GLuint i, GLenum param)
	{
		GLint value = 0;
		glGetVertexAttribiv(i, param, &value);
		return value;
	}
#endif

	void checkEnabled(GLuint vao, GLint lower, GLint upper)
	{
		for (int i = 0; i < Vcl::Graphics::OpenGL::GL::getInteger(GL_MAX_VERTEX_ATTRIBS); i++)
		{
			GLint enabled = vertexAttribiv(vao, i, GL_VERTEX_ATTRIB_ARRAY_ENABLED);
			EXPECT_EQ(i >= lower && i <= upper, enabled == 1);
		}
	}

	void checkSettings(GLuint vao, GLuint i, GLint ref_size, GLint ref_stride, GLint ref_type, GLint ref_norm, GLint ref_integral, GLint ref_div)
	{
		GLint size = vertexAttribiv(vao, i, GL_VERTEX_ATTRIB_ARRAY_SIZE);
		EXPECT_EQ(size, ref_size);
		GLint stride = vertexAttribiv(vao, i, GL_VERTEX_ATTRIB_ARRAY_STRIDE);
		EXPECT_EQ(stride, ref_stride);
		GLint type = vertexAttribiv(vao, i, GL_VERTEX_ATTRIB_ARRAY_TYPE);
		EXPECT_EQ(type, ref_type);
		GLint norm = vertexAttribiv(vao, i, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED);
		EXPECT_EQ(norm, ref_norm);
		GLint integral = vertexAttribiv(vao, i, GL_VERTEX_ATTRIB_ARRAY_INTEGER);
		EXPECT_EQ(integral, ref_integral);
		GLint divisor = vertexAttribiv(vao, i, GL_VERTEX_ATTRIB_ARRAY_DIVISOR);
		EXPECT_EQ(divisor, ref_div);
	}
}

using namespace Vcl::Graphics::Runtime::OpenGL;
using namespace Vcl::Graphics::Runtime;
using namespace Vcl::Graphics;

TEST(OpenGL, EmptyLayout)
{
	InputLayoutDescription in;
	InputLayout layout{ in };
	EXPECT_NE(0, layout.id());
}

TEST(OpenGL, Float4Layout)
{
	InputLayoutDescription in{
		{
			{ 0, sizeof(Eigen::Vector4f), VertexDataClassification::VertexDataPerObject },
		},
		{
			{ "Position", SurfaceFormat::R32G32B32A32_FLOAT, 0, 0, 0 },
		}
	};
	InputLayout layout{ in };
	EXPECT_NE(0, layout.id());

	layout.bind();

	checkEnabled(layout.id(), 0, 0);
	checkSettings(layout.id(), 0, 4, 0, GL_FLOAT, GL_FALSE, GL_FALSE, 0);
}

TEST(OpenGL, NormalizedSignedShort2Layout)
{
	InputLayoutDescription in{
		{
			{ 0, 2 * sizeof(short), VertexDataClassification::VertexDataPerInstance },
		},
		{
			{ "Position", SurfaceFormat::R16G16_SNORM, 0, 0, 0 },
		}
	};
	InputLayout layout{ in };
	EXPECT_NE(0, layout.id());

	layout.bind();

	checkEnabled(layout.id(), 0, 0);
	checkSettings(layout.id(), 0, 2, 0, GL_SHORT, GL_TRUE, GL_FALSE, 1);
}

TEST(OpenGL, NormalizedUnsignedShort4Layout)
{
	InputLayoutDescription in{
		{
			{ 0, 4 * sizeof(unsigned short), VertexDataClassification::VertexDataPerInstance },
		},
		{
			{ "Position", SurfaceFormat::R16G16B16A16_UNORM, 0, 0, 0 },
		}
	};
	InputLayout layout{ in };
	EXPECT_NE(0, layout.id());

	layout.bind();

	checkEnabled(layout.id(), 0, 0);
	checkSettings(layout.id(), 0, 4, 0, GL_UNSIGNED_SHORT, GL_TRUE, GL_FALSE, 1);
}

TEST(OpenGL, SignedByte1Layout)
{
	InputLayoutDescription in{
		{
			{ 0, sizeof(char), VertexDataClassification::VertexDataPerObject },
		},
		{
			{ "Position", SurfaceFormat::R8_SINT, 0, 0, 0 },
		}
	};
	InputLayout layout{ in };
	EXPECT_NE(0, layout.id());

	layout.bind();

	checkEnabled(layout.id(), 0, 0);
	checkSettings(layout.id(), 0, 1, 0, GL_BYTE, GL_FALSE, GL_TRUE, 0);
}
