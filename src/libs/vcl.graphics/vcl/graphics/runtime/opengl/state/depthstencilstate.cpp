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
#include <vcl/graphics/runtime/opengl/state/depthstencilstate.h>

// C++ standard library
#include <utility>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/opengl/gl.h>

#ifdef VCL_OPENGL_SUPPORT
namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	using namespace Vcl::Graphics::OpenGL;

	DepthStencilState::DepthStencilState(const DepthStencilDescription& desc)
	: _desc(desc)
	{
	}

	void DepthStencilState::bind()
	{
		// Enable depth buffer tests
		if (_desc.DepthEnable)
		{
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(toGLenum(desc().DepthFunc));
		}
		else
		{
			glDisable(GL_DEPTH_TEST);
		}

		// Configure the depth-buffer writing
		if (_desc.DepthWriteMask == DepthWriteMethod::All)
			glDepthMask(GL_TRUE);
		else
			glDepthMask(GL_FALSE);
	}

	void DepthStencilState::record(Graphics::OpenGL::CommandStream& states)
	{
		// Enable depth buffer tests
		if (desc().DepthEnable)
		{
			states.emplace(CommandType::Enable, GL_DEPTH_TEST);
			states.emplace(CommandType::DepthFunc, toGLenum(desc().DepthFunc));
		}

		// Configure the depth-buffer writing
		GLboolean depth_write = desc().DepthWriteMask == DepthWriteMethod::All ? GL_TRUE : GL_FALSE;
		states.emplace(CommandType::DepthMask, depth_write);
	}

	bool DepthStencilState::isValid() const
	{
		bool valid = true;

		if (desc().DepthEnable)
		{
			valid &= glIsEnabled(GL_DEPTH_TEST) == GL_TRUE;
			valid &= GL::getEnum(GL_DEPTH_FUNC) == toGLenum(desc().DepthFunc);
		}
		else
		{
			valid &= glIsEnabled(GL_DEPTH_TEST) == GL_FALSE;
		}

		valid &= GL::getEnum(GL_DEPTH_WRITEMASK) == (desc().DepthWriteMask == DepthWriteMethod::All);

		return valid;
	}

	GLenum DepthStencilState::toGLenum(ComparisonFunction op)
	{
		switch (op)
		{
		case ComparisonFunction::Never:        return GL_NEVER;
		case ComparisonFunction::Less:         return GL_LESS;
		case ComparisonFunction::Equal:        return GL_EQUAL;
		case ComparisonFunction::LessEqual:    return GL_LEQUAL;
		case ComparisonFunction::Greater:      return GL_GREATER;
		case ComparisonFunction::NotEqual:     return GL_NOTEQUAL;
		case ComparisonFunction::GreaterEqual: return GL_GEQUAL;
		case ComparisonFunction::Always:       return GL_ALWAYS;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_INVALID_ENUM;
	}

	GLenum DepthStencilState::toGLenum(StencilOperation op)
	{
		switch (op)
		{
		case StencilOperation::Keep:             return GL_KEEP;
		case StencilOperation::Zero:             return GL_ZERO;
		case StencilOperation::Replace:          return GL_REPLACE;
		case StencilOperation::IncreaseSaturate: return GL_INCR;
		case StencilOperation::DecreaseSaturate: return GL_DECR;
		case StencilOperation::Invert:           return GL_INVERT;
		case StencilOperation::IncreaseWrap:     return GL_INCR_WRAP;
		case StencilOperation::DecreaseWrap:     return GL_DECR_WRAP;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_INVALID_ENUM;
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
