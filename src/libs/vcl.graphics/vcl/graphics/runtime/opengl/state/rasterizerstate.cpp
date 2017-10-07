/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#include <vcl/graphics/runtime/opengl/state/rasterizerstate.h>

// C++ standard library
#include <utility>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/opengl/gl.h>

#ifdef VCL_OPENGL_SUPPORT
namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	using namespace Vcl::Graphics::OpenGL;

	RasterizerState::RasterizerState(const RasterizerDescription& desc)
	: _desc(desc)
	{
	}

	void RasterizerState::bind()
	{
		switch (desc().CullMode)
		{
		case CullMode::None:
		{
			glDisable(GL_CULL_FACE);
			break;
		}
		case CullMode::Front:
		{
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);

			if (desc().FrontCounterClockwise)
				glFrontFace(GL_CCW);
			else
				glFrontFace(GL_CW);
			
			break;
		}
		case CullMode::Back:
		{
			glEnable(GL_CULL_FACE);
			glCullFace(GL_BACK);

			if (desc().FrontCounterClockwise)
				glFrontFace(GL_CCW);
			else
				glFrontFace(GL_CW);
			
			break;
		}
		}

		// Configure fill mode
		glPolygonMode(GL_FRONT_AND_BACK, toGLenum(desc().FillMode));
	}
	
	void RasterizerState::record(Graphics::OpenGL::CommandStream& states)
	{
		switch (desc().CullMode)
		{
		case CullMode::None:
		{
			states.emplace(CommandType::Disable, GL_CULL_FACE);
			break;
		}
		case CullMode::Front:
		{
			states.emplace(CommandType::Enable, GL_CULL_FACE);
			states.emplace(CommandType::CullFace, GL_FRONT);

			if (desc().FrontCounterClockwise)
				states.emplace(CommandType::FrontFace, GL_CCW);
			else
				states.emplace(CommandType::FrontFace, GL_CW);
			
			break;
		}
		case CullMode::Back:
		{
			states.emplace(CommandType::Enable, GL_CULL_FACE);
			states.emplace(CommandType::CullFace, GL_BACK);

			if (desc().FrontCounterClockwise)
				states.emplace(CommandType::FrontFace, GL_CCW);
			else
				states.emplace(CommandType::FrontFace, GL_CW);
			
			break;
		}
		}

		// Configure fill mode
		states.emplace(CommandType::PolygonMode, GL_FRONT_AND_BACK, toGLenum(desc().FillMode));
	}

	bool RasterizerState::isValid() const
	{
		bool valid = true;
		
		if (desc().CullMode ==  CullMode::None)
		{
			valid &= glIsEnabled(GL_CULL_FACE) == GL_FALSE;
		}
		else
		{
			valid &= glIsEnabled(GL_CULL_FACE) == GL_TRUE;
			valid &= OpenGL::GL::getEnum(GL_CULL_FACE_MODE) == toGLenum(desc().CullMode);
			valid &= OpenGL::GL::getEnum(GL_FRONT_FACE) == (desc().FrontCounterClockwise ? GL_CCW : GL_CW);
		}

		GLint values[2];
		glGetIntegerv(GL_POLYGON_MODE, values);
		valid &= (values[0] == toGLenum(desc().FillMode) && values[1] == toGLenum(desc().FillMode));

		return valid;
	}

	bool RasterizerState::check() const
	{
		if (desc().CullMode ==  CullMode::None)
		{
			VclCheck(glIsEnabled(GL_CULL_FACE) == GL_FALSE, "Rasterstate is correct");
		}
		else
		{
			VclCheck(glIsEnabled(GL_CULL_FACE) == GL_TRUE, "Rasterstate is correct");
			VclCheck(OpenGL::GL::getEnum(GL_CULL_FACE_MODE) == toGLenum(desc().CullMode), "Rasterstate is correct");
			VclCheck(OpenGL::GL::getEnum(GL_FRONT_FACE) == (desc().FrontCounterClockwise ? GL_CCW : GL_CW), "Rasterstate is correct");
		}
		
		GLint values[2];
		glGetIntegerv(GL_POLYGON_MODE, values);
		VclCheck(values[0] == toGLenum(desc().FillMode) && values[1] == toGLenum(desc().FillMode), "Rasterstate is correct");

		return true;
	}

	GLenum RasterizerState::toGLenum(CullMode op)
	{
		switch (op)
		{
		case CullMode::None:  return GL_INVALID_ENUM;
		case CullMode::Front: return GL_FRONT;
		case CullMode::Back:  return GL_BACK;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_INVALID_ENUM;
	}

	GLenum RasterizerState::toGLenum(FillMode op)
	{
		switch (op)
		{
		case FillMode::Solid:     return GL_FILL;
		case FillMode::Wireframe: return GL_LINE;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_INVALID_ENUM;
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
