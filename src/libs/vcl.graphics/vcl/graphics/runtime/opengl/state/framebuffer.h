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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>
#include <vcl/config/opengl.h>

// C++ standard library
#include <array>

// VCL
#include <vcl/graphics/runtime/opengl/resource/resource.h>
#include <vcl/graphics/runtime/resource/texture.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	class Framebuffer : public Resource
	{
	public:
		Framebuffer
		(
			const Runtime::Texture** colourTargets, size_t nrColourTargets,
			const Runtime::Texture* depthTarget
		);
		Framebuffer(Framebuffer&&) = default;
		Framebuffer(const Framebuffer&) = delete;
		virtual ~Framebuffer();

		Framebuffer& operator= (Framebuffer&&) = default;
		Framebuffer& operator= (const Framebuffer&) = delete;
		
	public:
		void bind();
		void clear(int idx, const Eigen::Vector4f& colour);
		void clear(int idx, const Eigen::Vector4i& colour);
		void clear(int idx, const Eigen::Vector4ui& colour);
		void clear(float depth, int stencil);
		void clear(float depth);
		void clear(int stencil);

	private:
		bool checkGLFramebufferStatus(GLuint fbo);
		GLenum toDepthStencilAttachment(SurfaceFormat fmt);

	private:
		// Implementation defined maximum number of render targets
		static const unsigned int maxNrRenderTargets = 8;

		//! Colour render target
		std::array<const Runtime::Texture*, 8> _colourTargets;

		//! Depth target
		const Runtime::Texture* _depthTarget{ nullptr };
	};
}}}}
#endif // VCL_OPENGL_SUPPORT

