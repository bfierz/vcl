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
	const char* GL::getProfileInfo()
	{
		GLint profile;
		glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile);

		if (profile == GL_CONTEXT_CORE_PROFILE_BIT)
			return "Core";
		else if (profile == GL_CONTEXT_COMPATIBILITY_PROFILE_BIT)
			return "Compatibility";
		else
			return "Invalid";
	}

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
	
	AnyRenderType GL::toRenderType(SurfaceFormat fmt)
	{
		using namespace Vcl::Graphics::OpenGL;

		switch (fmt)
		{
		case SurfaceFormat::R32G32B32A32_FLOAT  : return RenderType<Eigen::Vector4f>();
		//case SurfaceFormat::R32G32B32A32_UINT   : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32B32A32_SINT   : return RenderType<Eigen::Vector4i>();
		//case SurfaceFormat::R16G16B16A16_FLOAT  : return RenderType<Eigen::Vector4f>();
		//case SurfaceFormat::R16G16B16A16_UNORM  : gl_format = GL_RGBA16; break;
		//case SurfaceFormat::R16G16B16A16_UINT   : return RenderType<Eigen::Vector4f>();
		//case SurfaceFormat::R16G16B16A16_SNORM  : gl_format = GL_RGBA16_SNORM; break;
		//case SurfaceFormat::R16G16B16A16_SINT   : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32B32_FLOAT     : return RenderType<Eigen::Vector3f>();
		//case SurfaceFormat::R32G32B32_UINT      : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32B32_SINT      : return RenderType<Eigen::Vector3i>();
		case SurfaceFormat::R32G32_FLOAT        : return RenderType<Eigen::Vector2f>();
		//case SurfaceFormat::R32G32_UINT         : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32_SINT         : return RenderType<Eigen::Vector2i>();
		//case SurfaceFormat::D32_FLOAT_S8X24_UINT: gl_format = GL_DEPTH32F_STENCIL8; break;
		//case SurfaceFormat::R10G10B10A2_UNORM   : gl_format = GL_RGB10_A2; break;
		//case SurfaceFormat::R10G10B10A2_UINT    : gl_format = GL_RGB10_A2UI; break;
		//case SurfaceFormat::R11G11B10_FLOAT     : gl_format = GL_R11F_G11F_B10F; break;
		//case SurfaceFormat::R8G8B8A8_UNORM      : gl_format = GL_RGBA8; break;
		//case SurfaceFormat::R8G8B8A8_UNORM_SRGB : gl_format = GL_SRGB8_ALPHA8; break;
		//case SurfaceFormat::R8G8B8A8_UINT       : gl_format = GL_RGBA8UI; break;
		//case SurfaceFormat::R8G8B8A8_SNORM      : gl_format = GL_RGBA8_SNORM; break;
		//case SurfaceFormat::R8G8B8A8_SINT       : gl_format = GL_RGBA8I; break;
		//case SurfaceFormat::R16G16_FLOAT        : gl_format = GL_RG16F; break;
		//case SurfaceFormat::R16G16_UNORM        : gl_format = GL_RG16; break;
		//case SurfaceFormat::R16G16_UINT         : gl_format = GL_RG16UI; break;
		//case SurfaceFormat::R16G16_SNORM        : gl_format = GL_RG16_SNORM; break;
		//case SurfaceFormat::R16G16_SINT         : gl_format = GL_RG16I; break;
		//case SurfaceFormat::D32_FLOAT           : gl_format = GL_DEPTH_COMPONENT32F; break;
		//case SurfaceFormat::R32_FLOAT           : gl_format = GL_R32F; break;
		//case SurfaceFormat::R32_UINT            : gl_format = GL_R32UI; break;
		//case SurfaceFormat::R32_SINT            : gl_format = GL_R32I; break;
		//case SurfaceFormat::D24_UNORM_S8_UINT   : gl_format = GL_DEPTH24_STENCIL8; break;
		//case SurfaceFormat::R8G8_UNORM          : gl_format = GL_RG8; break;
		//case SurfaceFormat::R8G8_UINT           : gl_format = GL_RG8UI; break;
		//case SurfaceFormat::R8G8_SNORM          : gl_format = GL_RG8_SNORM; break;
		//case SurfaceFormat::R8G8_SINT           : gl_format = GL_RG8I; break;
		//case SurfaceFormat::R16_FLOAT           : gl_format = GL_R16F; break;
		//case SurfaceFormat::D16_UNORM           : gl_format = GL_DEPTH_COMPONENT16; break;
		//case SurfaceFormat::R16_UNORM           : gl_format = GL_R16; break;
		//case SurfaceFormat::R16_UINT            : gl_format = GL_R16UI; break;
		//case SurfaceFormat::R16_SNORM           : gl_format = GL_R16_SNORM; break;
		//case SurfaceFormat::R16_SINT            : gl_format = GL_R16I; break;
		//case SurfaceFormat::R8_UNORM            : gl_format = GL_R8; break;
		//case SurfaceFormat::R8_UINT             : gl_format = GL_R8UI; break;
		//case SurfaceFormat::R8_SNORM            : gl_format = GL_R8_SNORM; break;
		//case SurfaceFormat::R8_SINT             : gl_format = GL_R8I; break;
		default: VclDebugError("Unsupported colour format.");
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
		switch(status)
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
