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
#include <vcl/graphics/runtime/framebuffer.h>

// VCL
#include <vcl/graphics/runtime/graphicsengine.h>

#ifdef VCL_OPENGL_SUPPORT
namespace Vcl { namespace Graphics { namespace Runtime {
	Framebuffer::Framebuffer(const FramebufferDescription& desc)
	: _desc{ desc }
	{
	}

	GBuffer::GBuffer(ref_ptr<GraphicsEngine> engine, const FramebufferDescription& desc)
	: Framebuffer{ desc }
	{
		// Create the depth buffer
		Texture2DDescription depth_desc;
		depth_desc.Width = desc.Width;
		depth_desc.Height = desc.Height;
		depth_desc.MipLevels = 1;
		depth_desc.ArraySize = 1;
		depth_desc.Format = desc.DepthBuffer.Format;

		_depthTarget = engine->createResource(depth_desc);

		// Create the render targets
		for (unsigned int i = 0; i < desc.NrRenderTargets; i++)
		{
			Texture2DDescription tex_desc;
			tex_desc.Width = desc.Width;
			tex_desc.Height = desc.Height;
			tex_desc.MipLevels = 1;
			tex_desc.ArraySize = 1;
			tex_desc.Format = desc.RenderTargets[i].Format;

			_renderTargets[i] = engine->createResource(tex_desc);
		}
	}

	void GBuffer::bind(ref_ptr<GraphicsEngine> engine)
	{
		std::array<ref_ptr<Texture>, 8> textures;
		for (size_t i = 0; i < description().NrRenderTargets; i++)
			textures[i] = _renderTargets[i];

		stdext::span<ref_ptr<Texture>> rt{ textures.data(), description().NrRenderTargets };
		engine->setRenderTargets(rt, _depthTarget);
		_engine = engine;
	}
	void GBuffer::clear(int idx, const Eigen::Vector4f& colour)
	{
		_engine->clear(idx, colour);
	}
	void GBuffer::clear(int idx, const Eigen::Vector4i& colour)
	{
		_engine->clear(idx, colour);
	}
	void GBuffer::clear(int idx, const Eigen::Vector4ui& colour)
	{
		_engine->clear(idx, colour);
	}
	void GBuffer::clear(float depth, int stencil)
	{
		_engine->clear(depth, stencil);
	}
	void GBuffer::clear(float depth)
	{
		_engine->clear(depth);
	}
	void GBuffer::clear(int stencil)
	{
		_engine->clear(stencil);
	}

	ABuffer::ABuffer(ref_ptr<GraphicsEngine> engine, const FramebufferDescription& desc)
	: Framebuffer{ desc }
	{
		VclRequire(desc.DepthBuffer.Format != SurfaceFormat::Unknown, "Depth-buffer is configured.");

		// Allocate the head-buffer
		BufferDescription headDesc;
		headDesc.Usage = BufferUsage::Storage;
		headDesc.SizeInBytes = (2 + desc.Width * desc.Height) * sizeof(uint32_t);

		_headBuffer = engine->createResource(headDesc);

		// Allocate the fragment pool
		BufferDescription poolDesc;
		poolDesc.Usage = BufferUsage::Storage;
		poolDesc.SizeInBytes = (1 + desc.Width * desc.Height) * 8 * (2 * sizeof(uint32_t));

		_headBuffer = engine->createResource(poolDesc);
	}

	void ABuffer::bind(ref_ptr<GraphicsEngine> engine)
	{
	}

	void ABuffer::clear(int idx, const Eigen::Vector4f& colour)
	{
	}
	void ABuffer::clear(int idx, const Eigen::Vector4i& colour)
	{
	}
	void ABuffer::clear(int idx, const Eigen::Vector4ui& colour)
	{
	}
	void ABuffer::clear(float depth, int stencil)
	{
	}
	void ABuffer::clear(float depth)
	{
	}
	void ABuffer::clear(int stencil)
	{
	}
	void ABuffer::resolve()
	{
	}
}}}
#endif // VCL_OPENGL_SUPPORT
