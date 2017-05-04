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
#include <vcl/graphics/runtime/opengl/state/framebuffer.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/graphics/runtime/opengl/resource/texture2darray.h>
#include <vcl/graphics/runtime/opengl/resource/texture3d.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Framebuffer::Framebuffer
	(
		const Runtime::Texture** colourTargets, size_t nrColourTargets,
		const Runtime::Texture* depthTarget
	)
	: _depthTarget{ depthTarget }
	{
		_colourTargets.assign(nullptr);

		// Create a framebuffer
		glCreateFramebuffers(1, &_glId);

		// Activate the necessary numbers of draw buffers
		std::vector<GLenum> colourAttachements(maxNrRenderTargets);
		for (unsigned int i = 0; i < static_cast<unsigned int>(colourAttachements.size()); i++)
			colourAttachements[i] = GL_COLOR_ATTACHMENT0 + i;
		glNamedFramebufferDrawBuffers(_glId, maxNrRenderTargets, colourAttachements.data());

		// Bind the render targets
		if (depthTarget)
		{
			switch (depthTarget->type())
			{
			case TextureType::Texture2D:
			{
				SurfaceFormat depth_fmt = depthTarget->format();
				GLenum depth_attachment_type = toDepthStencilAttachment(depth_fmt);
				
				auto& tex = static_cast<const Runtime::OpenGL::Texture2D&>(*depthTarget);
				glNamedFramebufferTexture(_glId, depth_attachment_type, tex.id(), 0);
				break;
			}
			case TextureType::Texture2DArray:
			{
				SurfaceFormat depth_fmt = depthTarget->format();
				GLenum depth_attachment_type = toDepthStencilAttachment(depth_fmt);

				auto& tex = static_cast<const Runtime::OpenGL::Texture2DArray&>(*depthTarget);
				if (tex.firstLayer() == 0 && tex.layers() > 1)
				{
					glNamedFramebufferTexture(_glId, depth_attachment_type, tex.id(), tex.firstMipMapLevel());
				}
				else if (tex.firstLayer() == 0 && tex.layers() == 1)
				{
					glNamedFramebufferTextureLayer(_glId, depth_attachment_type, tex.id(), tex.firstMipMapLevel(), tex.firstLayer());
				}
				else
				{
					VclDebugError("Combination supported.");
				}
			
				break;
			}
			default:
				VclDebugError("Not implemented.");
			}
		}
		else
		{
			glNamedFramebufferTexture(_glId, GL_DEPTH_ATTACHMENT, 0, 0);
		}
		
		for (size_t v = 0; v < nrColourTargets; v++)
		{
			_colourTargets[v] = colourTargets[v];

			auto view = colourTargets[v];
			switch (view->type())
			{
			case TextureType::Texture2D:
			{
				auto& tex = static_cast<const Runtime::OpenGL::Texture2D&>(*view);
				glNamedFramebufferTexture(_glId, GL_COLOR_ATTACHMENT0 + v, tex.id(), view->firstMipMapLevel());
				break;
			}
			case TextureType::Texture2DArray:
			{
				auto& tex = static_cast<const Runtime::OpenGL::Texture2DArray&>(*view);
				int first = (int) tex.firstLayer();
				int size  = (int) tex.layers();
				
				if (first == 0 && size == tex.layers())
				{
					glNamedFramebufferTexture(_glId, GL_COLOR_ATTACHMENT0 + v, tex.id(), tex.firstMipMapLevel());
				}
				else if (size == 1)
				{
					glNamedFramebufferTextureLayer(_glId, GL_COLOR_ATTACHMENT0 + v, tex.id(), tex.firstMipMapLevel(), first);
				}
				else
				{
					VclDebugError("Not available in OpenGL");
				}
				
				break;
			}
			case TextureType::Texture3D:
			{
				auto& tex = static_cast<const Runtime::OpenGL::Texture3D&>(*view);
				//int first = (int) tex.firstDepthSlice();
				//int size = (int) tex.depthSlices();
				int first = 0;
				int size = (int) tex.depth();

				if (first == 0 && size >= tex.depth())
				{
					glNamedFramebufferTexture(_glId, GL_COLOR_ATTACHMENT0 + v, tex.id(), tex.firstMipMapLevel());
				}
				else if (size == 1)
				{
					glNamedFramebufferTextureLayer(_glId, GL_COLOR_ATTACHMENT0 + v, tex.id(), tex.firstMipMapLevel(), first);
				}
				else
				{
					VclDebugError("Not available in OpenGL");
				}
			
				break;
			}
			default:
				VclDebugError("Not implemented.");
			}
		}

		// Post condition
		VclEnsure(checkGLFramebufferStatus(_glId), "Framebuffer object is complete.");
	}

	Framebuffer::~Framebuffer()
	{
		// Delete FBO; bound targets are implicitly unbound
		if (_glId > 0)
			glDeleteFramebuffers(1, &_glId);
	}

	void Framebuffer::bind()
	{
		VclRequire(checkGLFramebufferStatus(_glId), "Framebuffer object is complete.");

		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _glId);
	}

	void Framebuffer::clear(int idx, const Eigen::Vector4f& colour)
	{
		VclRequire(checkGLFramebufferStatus(_glId), "Framebuffer object is complete.");

		glClearNamedFramebufferfv(_glId, GL_COLOR, idx, colour.data());
	}
	void Framebuffer::clear(int idx, const Eigen::Vector4i& colour)
	{
		VclRequire(checkGLFramebufferStatus(_glId), "Framebuffer object is complete.");

		glClearNamedFramebufferiv(_glId, GL_COLOR, idx, colour.data());
	}
	void Framebuffer::clear(int idx, const Eigen::Vector4ui& colour)
	{
		VclRequire(checkGLFramebufferStatus(_glId), "Framebuffer object is complete.");

		glClearNamedFramebufferuiv(_glId, GL_COLOR, idx, colour.data());
	}
	void Framebuffer::clear(float depth, int stencil)
	{
		VclRequire(checkGLFramebufferStatus(_glId), "Framebuffer object is complete.");

		glClearNamedFramebufferfi(_glId, GL_DEPTH_STENCIL, 0, depth, stencil);
	}
	void Framebuffer::clear(float depth)
	{
		VclRequire(checkGLFramebufferStatus(_glId), "Framebuffer object is complete.");

		glClearNamedFramebufferfv(_glId, GL_DEPTH, 0, &depth);
	}
	void Framebuffer::clear(int stencil)
	{
		VclRequire(checkGLFramebufferStatus(_glId), "Framebuffer object is complete.");

		glClearNamedFramebufferiv(_glId, GL_STENCIL, 0, &stencil);
	}

	bool Framebuffer::checkGLFramebufferStatus(GLuint fbo)
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

	GLenum Framebuffer::toDepthStencilAttachment(SurfaceFormat fmt)
	{
		switch (fmt)
		{
		case SurfaceFormat::Unknown:
			return GL_NONE;
		case SurfaceFormat::D16_UNORM:
			return GL_DEPTH_ATTACHMENT;
		case SurfaceFormat::D24_UNORM_S8_UINT:
			return GL_DEPTH_STENCIL_ATTACHMENT;
		case SurfaceFormat::D32_FLOAT:
			return GL_DEPTH_ATTACHMENT;
		case SurfaceFormat::D32_FLOAT_S8X24_UINT:
			return GL_DEPTH_STENCIL_ATTACHMENT;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}
	}
}}}}
