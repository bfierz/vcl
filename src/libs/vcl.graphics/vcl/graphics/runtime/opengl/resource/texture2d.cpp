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
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL {
	Texture2D::Texture2D(
		const Texture2DDescription& desc,
		const TextureResource* init_data)
	{
		initializeView(
			TextureType::Texture2D, desc.Format, desc.Usage,
			0, desc.MipLevels,
			0, 1,
			desc.Width, desc.Height, 1);
		initialise(init_data);
	}

	Texture2D::Texture2D(Texture2D&& rhs)
	: Texture(std::move(rhs))
	{
	}

	Texture2D::Texture2D(const Texture2D& rhs)
	: Texture(rhs)
	{
		initialise(nullptr);
	}

	std::unique_ptr<Runtime::Texture> Texture2D::clone() const
	{
		return std::make_unique<Texture2D>(*this);
	}

	void Texture2D::allocImpl(GLenum colour_fmt)
	{
#if defined(VCL_GL_ARB_direct_state_access)
		glTextureStorage2D(_glId, mipMapLevels(), colour_fmt, width(), height());
#elif defined(VCL_GL_EXT_direct_state_access)
		glTextureStorage2DEXT(_glId, GL_TEXTURE_2D, mipMapLevels(), colour_fmt, width(), height());
#else
		glTexStorage2D(GL_TEXTURE_2D, mipMapLevels(), colour_fmt, width(), height());
#endif
	}

	void Texture2D::updateImpl(const TextureResource& data)
	{
		ImageFormat img_fmt = toImageFormat(data.Format != SurfaceFormat::Unknown ? data.Format : format());
		GLsizei w = (GLsizei)data.Width;
		GLsizei h = (GLsizei)data.Height;
		GLsizei mip = (GLsizei)data.MipMap;

#if defined(VCL_GL_ARB_direct_state_access)
		glTextureSubImage2D(_glId, mip, 0, 0, w, h, img_fmt.Format, img_fmt.Type, data.data());
#elif defined(VCL_GL_EXT_direct_state_access)
		glTextureSubImage2DEXT(_glId, GL_TEXTURE_2D, mip, 0, 0, w, h, img_fmt.Format, img_fmt.Type, data.data());
#else
		glTexSubImage2D(GL_TEXTURE_2D, mip, 0, 0, w, h, img_fmt.Format, img_fmt.Type, data.data());
#endif
	}
}}}}
