/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#define VCL_GRAPHICS_HELIOS_OPENGL_INPUTLAYOUT_INST
#include <vcl/graphics/runtime/opengl/state/inputlayout.h>

// VCL
#include <vcl/graphics/opengl/gl.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	template<>
	struct VertexDataTypeTrait<float>
	{
		static const GLenum Type = GL_FLOAT;
		static const GLint Size = sizeof(float);
		static const GLint NrComponents = 1;
		static const bool IsIntegral = false;
	};

	template<>
	struct VertexDataTypeTrait<Eigen::Vector2f>
	{
		static const GLenum Type = GL_FLOAT;
		static const GLint Size = sizeof(Eigen::Vector2f::Scalar);
		static const GLint NrComponents = 2;
		static const bool IsIntegral = false;
	};

	template<>
	struct VertexDataTypeTrait<Eigen::Vector3f>
	{
		static const GLenum Type = GL_FLOAT;
		static const GLint Size = sizeof(Eigen::Vector3f::Scalar);
		static const GLint NrComponents = 3;
		static const bool IsIntegral = false;
	};
	
	template<>
	struct VertexDataTypeTrait<Eigen::Vector4f>
	{
		static const GLenum Type = GL_FLOAT;
		static const GLint Size = sizeof(Eigen::Vector4f::Scalar);
		static const GLint NrComponents = 4;
		static const bool IsIntegral = false;
	};
	
	template<>
	struct VertexDataTypeTrait<int>
	{
		static const GLenum Type = GL_INT;
		static const GLint Size = sizeof(int);
		static const GLint NrComponents = 1;
		static const bool IsIntegral = true;
	};
	
	template<>
	struct VertexDataTypeTrait<Eigen::Vector2i>
	{
		static const GLenum Type = GL_INT;
		static const GLint Size = sizeof(Eigen::Vector2i::Scalar);
		static const GLint NrComponents = 2;
		static const bool IsIntegral = true;
	};
	
	template<>
	struct VertexDataTypeTrait<Eigen::Vector3i>
	{
		static const GLenum Type = GL_INT;
		static const GLint Size = sizeof(Eigen::Vector3i::Scalar);
		static const GLint NrComponents = 3;
		static const bool IsIntegral = true;
	};
	
	template<>
	struct VertexDataTypeTrait<Eigen::Vector4i>
	{
		static const GLenum Type = GL_INT;
		static const GLint Size = sizeof(Eigen::Vector4i::Scalar);
		static const GLint NrComponents = 4;
		static const bool IsIntegral = true;
	};
}}}}

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	InputLayout::InputLayout(const Runtime::InputLayoutDescription& desc)
	{
		setup(desc);
	}

	InputLayout::~InputLayout()
	{
		glDeleteVertexArrays(1, &_vaoID);
	}

	void InputLayout::bind()
	{
		glBindVertexArray(_vaoID);
	}

	void InputLayout::setup(const InputLayoutDescription& desc)
	{
		Require(_vaoID == 0, "No yet created.");
		Require(glewIsSupported("GL_ARB_vertex_attrib_binding"), "Vertex attribute binding is supported.");

		glCreateVertexArrays(1, &_vaoID);

		int idx = 0;
		for (const auto& elem : desc)
		{
			Check(implies(elem.StreamType == VertexDataClassification::VertexDataPerInstance, elem.StepRate > 0 && elem.StepRate != -1), "Step rate is > 0 for per instance data.");

			// GL relevant enumerations
			auto rt = Vcl::Graphics::OpenGL::GL::toRenderType(elem.Format);
			
			for (int sub_loc = 0; sub_loc < std::max(1, (int) elem.NumberLocations); sub_loc++)
			{
				// Shader attribute location
				int loc = desc.location(idx) + sub_loc;

				// Enable the attribute 'loc'
				glEnableVertexArrayAttrib(_vaoID, loc);

				// Bind 'loc' to the vertex buffer 'elem.StreamIndex'
				glVertexArrayAttribBinding(_vaoID, loc, elem.InputSlot);

				// Configure the stream update rate
				if (elem.StreamType == Runtime::VertexDataClassification::VertexDataPerObject)
				{
					glVertexArrayBindingDivisor(_vaoID, elem.InputSlot, 0);
				}
				else if (elem.StreamType == Runtime::VertexDataClassification::VertexDataPerInstance)
				{
					glVertexArrayBindingDivisor(_vaoID, elem.InputSlot, elem.StepRate);
				}

				// Set which underlying number type is used
				GLuint elementOffset = elem.Offset + sub_loc * rt.componentSize() * rt.nrComponents();
				if (rt.isIntegral())
				{
					glVertexArrayAttribIFormat(_vaoID, loc, rt.nrComponents(), rt.componentType(), elementOffset);
				}
				else
				{
					glVertexArrayAttribFormat(_vaoID, loc, rt.nrComponents(), rt.componentType(), GL_FALSE, elementOffset);
				}
			}

			idx++;
		}
	}
}}}}
