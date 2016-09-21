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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <array>

// GSL

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/graphics/runtime/resource/buffer.h>
#include <vcl/graphics/runtime/resource/texture.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	class GraphicsEngine;

	struct RenderTargetDescription
	{
		SurfaceFormat Format{ SurfaceFormat::Unknown };
	};

	struct DepthBufferDescription
	{
		SurfaceFormat Format{ SurfaceFormat::Unknown };
	};
	struct FramebufferDescription
	{
		uint32_t Width;
		uint32_t Height;
		uint32_t NrRenderTargets;
		std::array<RenderTargetDescription, 8> RenderTargets;
		DepthBufferDescription DepthBuffer;
	};

	/*!
	 *	\brief Base class for frame buffer objects
	 *
	 *	Provides a generic interface.
	 */
	class Framebuffer
	{
	public:
		Framebuffer(const FramebufferDescription& desc);

	public:
		virtual void bind(GraphicsEngine* engine) = 0;
		virtual void clear(int idx, const Eigen::Vector4f& colour) = 0;
		virtual void clear(int idx, const Eigen::Vector4i& colour) = 0;
		virtual void clear(int idx, const Eigen::Vector4ui& colour) = 0;
		virtual void clear(float depth, int stencil) = 0;
		virtual void clear(float depth) = 0;
		virtual void clear(int stencil) = 0;

	public:
		const FramebufferDescription& description() const { return _desc; }

	public:
		uint32_t width()  const { return _desc.Width; }
		uint32_t height() const { return _desc.Height; }

	private:
		//! Description of the framebuffer
		FramebufferDescription _desc;
	};

	class GBuffer : public Framebuffer
	{
	public:
		GBuffer(const FramebufferDescription& desc);

	public:
		void bind(GraphicsEngine* engine) override;
		void clear(int idx, const Eigen::Vector4f& colour) override;
		void clear(int idx, const Eigen::Vector4i& colour) override;
		void clear(int idx, const Eigen::Vector4ui& colour) override;
		void clear(float depth, int stencil) override;
		void clear(float depth) override;
		void clear(int stencil) override;

		const Vcl::Graphics::Runtime::Texture& renderTarget(size_t idx) { return *_renderTargets[idx]; }

	private:
		//! Depth target
		owner_ptr<Vcl::Graphics::Runtime::Texture> _depthTarget;

		//! Data targets
		std::array<owner_ptr<Vcl::Graphics::Runtime::Texture>, 8> _renderTargets;
	};

	class ABuffer : public Framebuffer
	{
	public:
		ABuffer(const FramebufferDescription& desc);

	public:
		void bind(GraphicsEngine* engine) override;
		void clear(int idx, const Eigen::Vector4f& colour) override;
		void clear(int idx, const Eigen::Vector4i& colour) override;
		void clear(int idx, const Eigen::Vector4ui& colour) override;
		void clear(float depth, int stencil) override;
		void clear(float depth) override;
		void clear(int stencil) override;
		void resolve();

	private:
		//! The link to the first fragment
		owner_ptr<Vcl::Graphics::Runtime::Buffer> _headBuffer;
		
		//! Data targets
		std::array<owner_ptr<Vcl::Graphics::Runtime::Texture>, 8> _fragmentPools;
	};
}}}
