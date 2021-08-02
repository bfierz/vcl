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

#if defined(VCL_GL_ARB_direct_state_access)
#	define glCreateVertexArraysVCL glCreateVertexArrays
#	define glEnableVertexArrayAttribVCL glEnableVertexArrayAttrib
#	define glVertexArrayAttribBindingVCL glVertexArrayAttribBinding
#	define glVertexArrayBindingDivisorVCL glVertexArrayBindingDivisor
#	define glVertexArrayAttribIFormatVCL glVertexArrayAttribIFormat
#	define glVertexArrayAttribFormatVCL glVertexArrayAttribFormat
#elif defined(VCL_GL_EXT_direct_state_access)
#	define glCreateVertexArraysVCL glGenVertexArrays
#	define glEnableVertexArrayAttribVCL glEnableVertexArrayAttribEXT
#	define glVertexArrayAttribBindingVCL glVertexArrayVertexAttribBindingEXT
#	define glVertexArrayBindingDivisorVCL glVertexArrayVertexBindingDivisorEXT
#	define glVertexArrayAttribIFormatVCL glVertexArrayVertexAttribIFormatEXT
#	define glVertexArrayAttribFormatVCL glVertexArrayVertexAttribFormatEXT
#else
#	define glCreateVertexArraysVCL glGenVertexArrays
#	define glEnableVertexArrayAttribVCL(idx, loc) glEnableVertexAttribArray(loc)
#	define glVertexArrayAttribBindingVCL(idx, loc, slot) glVertexAttribBinding(loc, slot)
#	define glVertexArrayBindingDivisorVCL(idx, loc, divisor) glVertexBindingDivisor(loc, divisor)
#	define glVertexArrayAttribIFormatVCL(idx, loc, size, type, offset) glVertexAttribIFormat(loc, size, type, offset)
#	define glVertexArrayAttribFormatVCL(idx, loc, size, type, normalized, offset) glVertexAttribFormat(loc, size, type, normalized, offset)
#endif

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL {
	InputLayout::InputLayout(const Runtime::InputLayoutDescription& desc)
	: _desc(desc)
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
		VclRequire(_vaoID == 0, "No yet created.");
		VclRequire(glewIsSupported("GL_ARB_vertex_array_object"), "Vertex array objects are supported.");
		VclRequire(glewIsSupported("GL_ARB_vertex_attrib_binding"), "Vertex attribute binding is supported.");

		glCreateVertexArraysVCL(1, &_vaoID);
#if !(defined(VCL_GL_ARB_direct_state_access) || defined(VCL_GL_EXT_direct_state_access))
		glBindVertexArray(_vaoID);
#endif

		int idx = 0;
		for (const auto& elem : desc.attributes())
		{
			const auto& binding = desc.binding(elem.InputSlot);

			// GL relevant enumerations
			auto rt = Vcl::Graphics::OpenGL::GL::toRenderType(elem.Format);
			
			for (int sub_loc = 0; sub_loc < std::max(1, (int) elem.NumberLocations); sub_loc++)
			{
				// Shader attribute location
				int loc = desc.location(idx) + sub_loc;

				// Enable the attribute 'loc'
				glEnableVertexArrayAttribVCL(_vaoID, loc);

				// Bind 'loc' to the vertex buffer 'elem.StreamIndex'
				glVertexArrayAttribBindingVCL(_vaoID, loc, elem.InputSlot);

				// Configure the stream update rate
				if (binding.InputRate == Runtime::VertexDataClassification::VertexDataPerObject)
				{
					glVertexArrayBindingDivisorVCL(_vaoID, elem.InputSlot, 0);
				}
				else if (binding.InputRate == Runtime::VertexDataClassification::VertexDataPerInstance)
				{
					glVertexArrayBindingDivisorVCL(_vaoID, elem.InputSlot, 1);
				}

				// Set which underlying number type is used
				GLuint elementOffset = elem.Offset + sub_loc * rt.componentSize() * rt.nrComponents();
				if (rt.isIntegral())
				{
					glVertexArrayAttribIFormatVCL(_vaoID, loc, rt.nrComponents(), rt.componentType(), elementOffset);
				}
				else
				{
					GLboolean normalized = !(rt.componentType() == GL_FLOAT || rt.componentType() == GL_HALF_FLOAT);
					glVertexArrayAttribFormatVCL(_vaoID, loc, rt.nrComponents(), rt.componentType(), normalized, elementOffset);
				}
			}

			idx++;
		}
#if !(defined(VCL_GL_ARB_direct_state_access) || defined(VCL_GL_EXT_direct_state_access))
		glBindVertexArray(GL_NONE);
#endif
	}
}}}}
