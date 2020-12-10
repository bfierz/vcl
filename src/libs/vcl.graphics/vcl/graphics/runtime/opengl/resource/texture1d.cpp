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
#include <vcl/graphics/runtime/opengl/resource/texture1d.h>

#ifdef VCL_OPENGL_SUPPORT

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Texture1D::Texture1D
	(
		const Texture1DDescription& desc,
		const TextureResource* init_data
	)
	{
		initializeView
		(
			TextureType::Texture1D, desc.Format, desc.Usage,
			0, desc.MipLevels,
			0, 1,
			desc.Width, 1, 1
		);
		initialise(init_data);
	}

	std::unique_ptr<Runtime::Texture> Texture1D::clone() const
	{
		return std::make_unique<Texture1D>(*this);
	}

	void Texture1D::allocImpl(GLenum colour_fmt)
	{
#	if defined(VCL_GL_ARB_direct_state_access)
		glTextureStorage1D(_glId, mipMapLevels(), colour_fmt, width());
#	elif defined(VCL_GL_EXT_direct_state_access)
		glTextureStorage1DEXT(_glId, GL_TEXTURE_1D, mipMapLevels(), colour_fmt, width());
#	else
		glTexStorage1D(GL_TEXTURE_1D, mipMapLevels(), colour_fmt, width());
#	endif
	}

	void Texture1D::updateImpl(const TextureResource& data)
	{
		ImageFormat img_fmt = toImageFormat(data.Format != SurfaceFormat::Unknown ? data.Format : format());
		GLsizei w = (GLsizei)data.Width;
		GLsizei mip = (GLsizei)data.MipMap;

#	if defined(VCL_GL_ARB_direct_state_access)
		glTextureSubImage1D(_glId, mip, 0, w, img_fmt.Format, img_fmt.Type, data.data());
#	elif defined(VCL_GL_EXT_direct_state_access)
		glTextureSubImage1DEXT(_glId, GL_TEXTURE_1D, mip, 0, w, img_fmt.Format, img_fmt.Type, data.data());
#	else
		glTexSubImage1D(GL_TEXTURE_1D, mip, 0, w, img_fmt.Format, img_fmt.Type, data.data());
#	endif
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
