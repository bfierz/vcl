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
#include <vcl/graphics/opengl/gl.h>

// VCL
#include <vcl/core/contract.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace OpenGL
{
	GLenum GL::getEnum(GLenum e)
	{
		GLint val = getInteger(e);

		return (GLenum) val;
	}
	
	GLenum GL::getEnum(GLenum e, int i)
	{
		GLint val = getInteger(e, i);

		return (GLenum) val;
	}

	GLint GL::getInteger(GLenum e)
	{
		GLint val;
		glGetIntegerv(e, &val);

		return val;
	}

	GLint GL::getInteger(GLenum e, int i)
	{
		GLint val;
		glGetIntegeri_v(e, i, &val);

		return val;
	}

	bool GL::checkGLError()
	{
		// Get the OpenGL error
		GLenum err = glGetError();

		switch (err)
		{
		case GL_NO_ERROR:
			return true;

		case GL_INVALID_ENUM:
			DebugError("Given when an enumeration parameter contains an enum that is not allowed for that function.");
			return false;
		case GL_INVALID_VALUE:
			DebugError("Given when a numerical parameter does not conform to the range requirements that the function places upon it.");
			return false;
		case GL_INVALID_OPERATION:
			DebugError("Given when the function in question cannot be executed because of state that has been set in the context.");
			return false;
		case GL_STACK_OVERFLOW:
			DebugError("Given when a stack pushing operation causes a stack to overflow the limit of that stack's size.");
			return false;
		case GL_STACK_UNDERFLOW:
			DebugError("Given when a stack popping operation is given when the stack is already at its lowest point.");
			return false;
		case GL_OUT_OF_MEMORY:
			DebugError("Given when performing an operation that can allocate memory, when the memory in question cannot be allocated.");
			return false;
		case GL_TABLE_TOO_LARGE:
			DebugError("Given if the optional imaging subset (GL_ARB_imaging) is supported.");
			return false;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			DebugError("Given if an operation on a framebuffer threw the error.");
			return false;

		default:
			DebugError("[ERROR] Unknow error.");
			return false;
		}
	}
	
	bool GL::checkGLFramebufferStatus(GLuint fbo)
	{
		// check FBO status
		GLenum status = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);   
		switch(status)
		{
		case GL_FRAMEBUFFER_COMPLETE:
			return true;

		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			DebugError("[ERROR] Framebuffer incomplete: Attachment is NOT complete.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			DebugError("[ERROR] Framebuffer incomplete: No image is attached to FBO.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			DebugError("[ERROR] Framebuffer incomplete: Draw buffer.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			DebugError("[ERROR] Framebuffer incomplete: Read buffer.");
			return false;

		case GL_FRAMEBUFFER_UNSUPPORTED:
			DebugError("[ERROR] Unsupported by FBO implementation.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			DebugError("[ERROR] Framebuffer incomplete: Multisample.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			DebugError("[ERROR] Framebuffer incomplete: Layer target is NOT complete.");
			return false;

		case GL_FRAMEBUFFER_UNDEFINED:
		default:
			DebugError("[ERROR] Unknow error.");
			return false;
		}
	}
}}}
#endif // VCL_OPENGL_SUPPORT
