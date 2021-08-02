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

namespace Vcl { namespace Graphics { namespace OpenGL {
	const char* GL::getProfileInfo()
	{
		GLint profile;
		glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile);

		if (profile == GL_CONTEXT_CORE_PROFILE_BIT)
			return "Core";
		else if (profile == GL_CONTEXT_COMPATIBILITY_PROFILE_BIT)
			return "Compatibility";
		else if (profile == 0x4) // GLX_CONTEXT_ES_PROFILE_BIT_EXT/WGL_CONTEXT_ES_PROFILE_BIT_EXT
			return "Embedded";
		else
			return "Invalid";
	}

	GLenum GL::getEnum(GLenum e)
	{
		GLint val = getInteger(e);

		return (GLenum)val;
	}

	GLenum GL::getEnum(GLenum e, int i)
	{
		GLint val = getInteger(e, i);

		return (GLenum)val;
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

	AnyRenderType GL::toRenderType(SurfaceFormat fmt)
	{
		using namespace Vcl::Graphics::OpenGL;

		switch (fmt)
		{
		case SurfaceFormat::R32G32B32A32_FLOAT  : return RenderType<Float4>();
		case SurfaceFormat::R32G32B32A32_UINT   : return RenderType<UnsignedInt4>();
		case SurfaceFormat::R32G32B32A32_SINT   : return RenderType<SignedInt4>();
		case SurfaceFormat::R16G16B16A16_FLOAT  : return RenderType<Half4>();
		case SurfaceFormat::R16G16B16A16_UNORM  : return RenderType<NormalizedUnsignedShort4>();
		case SurfaceFormat::R16G16B16A16_UINT   : return RenderType<UnsignedShort4>();
		case SurfaceFormat::R16G16B16A16_SNORM  : return RenderType<NormalizedSignedShort4>();
		case SurfaceFormat::R16G16B16A16_SINT   : return RenderType<SignedShort4>();
		case SurfaceFormat::R32G32B32_FLOAT     : return RenderType<Float3>();
		case SurfaceFormat::R32G32B32_UINT      : return RenderType<UnsignedInt3>();
		case SurfaceFormat::R32G32B32_SINT      : return RenderType<SignedInt3>();
		case SurfaceFormat::R32G32_FLOAT        : return RenderType<Float2>();
		case SurfaceFormat::R32G32_UINT         : return RenderType<UnsignedInt2>();
		case SurfaceFormat::R32G32_SINT         : return RenderType<SignedInt2>();
		//case SurfaceFormat::D32_FLOAT_S8X24_UINT: return RenderType<>();
		//case SurfaceFormat::R10G10B10A2_UNORM   : return RenderType<>();
		//case SurfaceFormat::R10G10B10A2_UINT    : return RenderType<>();
		//case SurfaceFormat::R11G11B10_FLOAT     : return RenderType<>();
		case SurfaceFormat::R8G8B8A8_UNORM      : return RenderType<NormalizedUnsignedByte4>();
		//case SurfaceFormat::R8G8B8A8_UNORM_SRGB : return RenderType<>();
		case SurfaceFormat::R8G8B8A8_UINT       : return RenderType<UnsignedByte4>();
		case SurfaceFormat::R8G8B8A8_SNORM      : return RenderType<NormalizedSignedByte4>();
		case SurfaceFormat::R8G8B8A8_SINT       : return RenderType<SignedByte4>();
		case SurfaceFormat::R16G16_FLOAT        : return RenderType<Half2>();
		case SurfaceFormat::R16G16_UNORM        : return RenderType<NormalizedUnsignedShort2>();
		case SurfaceFormat::R16G16_UINT         : return RenderType<UnsignedShort2>();
		case SurfaceFormat::R16G16_SNORM        : return RenderType<NormalizedSignedShort2>();
		case SurfaceFormat::R16G16_SINT         : return RenderType<SignedShort2>();
		//case SurfaceFormat::D32_FLOAT           : return RenderType<>();
		case SurfaceFormat::R32_FLOAT           : return RenderType<Float>();
		case SurfaceFormat::R32_UINT            : return RenderType<UnsignedInt>();
		case SurfaceFormat::R32_SINT            : return RenderType<SignedInt>();
		//case SurfaceFormat::D24_UNORM_S8_UINT   : return RenderType<>();
		case SurfaceFormat::R8G8_UNORM          : return RenderType<NormalizedUnsignedByte2>();
		case SurfaceFormat::R8G8_UINT           : return RenderType<UnsignedByte2>();
		case SurfaceFormat::R8G8_SNORM          : return RenderType<NormalizedSignedByte2>();
		case SurfaceFormat::R8G8_SINT           : return RenderType<SignedByte2>();
		case SurfaceFormat::R16_FLOAT           : return RenderType<Half>();
		//case SurfaceFormat::D16_UNORM           : return RenderType<>();
		case SurfaceFormat::R16_UNORM           : return RenderType<NormalizedUnsignedShort>();
		case SurfaceFormat::R16_UINT            : return RenderType<UnsignedShort>();
		case SurfaceFormat::R16_SNORM           : return RenderType<NormalizedSignedShort>();
		case SurfaceFormat::R16_SINT            : return RenderType<SignedShort>();
		case SurfaceFormat::R8_UNORM            : return RenderType<NormalizedUnsignedByte>();
		case SurfaceFormat::R8_UINT             : return RenderType<UnsignedByte>();
		case SurfaceFormat::R8_SNORM            : return RenderType<NormalizedSignedByte>();
		case SurfaceFormat::R8_SINT             : return RenderType<SignedByte>();
		default: VclDebugError("Unsupported surface format.");
		};

		return RenderType<void>();
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
			VclDebugError("Given when an enumeration parameter contains an enum that is not allowed for that function.");
			return false;
		case GL_INVALID_VALUE:
			VclDebugError("Given when a numerical parameter does not conform to the range requirements that the function places upon it.");
			return false;
		case GL_INVALID_OPERATION:
			VclDebugError("Given when the function in question cannot be executed because of state that has been set in the context.");
			return false;
		case GL_STACK_OVERFLOW:
			VclDebugError("Given when a stack pushing operation causes a stack to overflow the limit of that stack's size.");
			return false;
		case GL_STACK_UNDERFLOW:
			VclDebugError("Given when a stack popping operation is given when the stack is already at its lowest point.");
			return false;
		case GL_OUT_OF_MEMORY:
			VclDebugError("Given when performing an operation that can allocate memory, when the memory in question cannot be allocated.");
			return false;
		case GL_TABLE_TOO_LARGE:
			VclDebugError("Given if the optional imaging subset (GL_ARB_imaging) is supported.");
			return false;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			VclDebugError("Given if an operation on a framebuffer threw the error.");
			return false;

		default:
			VclDebugError("[ERROR] Unknow error.");
			return false;
		}
	}

	bool GL::checkGLFramebufferStatus(GLuint fbo)
	{
		// check FBO status
		GLenum status = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
		switch (status)
		{
		case GL_FRAMEBUFFER_COMPLETE:
			return true;

		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			VclDebugError("[ERROR] Framebuffer incomplete: Attachment is NOT complete.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			VclDebugError("[ERROR] Framebuffer incomplete: No image is attached to FBO.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			VclDebugError("[ERROR] Framebuffer incomplete: Draw buffer.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			VclDebugError("[ERROR] Framebuffer incomplete: Read buffer.");
			return false;

		case GL_FRAMEBUFFER_UNSUPPORTED:
			VclDebugError("[ERROR] Unsupported by FBO implementation.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			VclDebugError("[ERROR] Framebuffer incomplete: Multisample.");
			return false;

		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			VclDebugError("[ERROR] Framebuffer incomplete: Layer target is NOT complete.");
			return false;

		case GL_FRAMEBUFFER_UNDEFINED:
		default:
			VclDebugError("[ERROR] Unknow error.");
			return false;
		}
	}
}}}
#endif // VCL_OPENGL_SUPPORT
