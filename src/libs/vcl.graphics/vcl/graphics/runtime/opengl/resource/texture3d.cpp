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
#include <vcl/graphics/runtime/opengl/resource/texture3d.h>

#ifdef VCL_OPENGL_SUPPORT

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL {
	Texture3D::Texture3D(
		const Texture3DDescription& desc,
		const TextureResource* init_data)
	{
		initializeView(
			TextureType::Texture3D, desc.Format, desc.Usage,
			0, desc.MipLevels,
			0, 1,
			desc.Width, desc.Height, desc.Depth);
		initialise(init_data);
	}

	std::unique_ptr<Runtime::Texture> Texture3D::clone() const
	{
		return std::make_unique<Texture3D>(*this);
	}

	void Texture3D::allocImpl(GLenum colour_fmt)
	{
#	if defined(VCL_GL_ARB_direct_state_access)
		glTextureStorage3D(_glId, mipMapLevels(), colour_fmt, width(), height(), depth());
#	elif defined(VCL_GL_EXT_direct_state_access)
		glTextureStorage3DEXT(_glId, GL_TEXTURE_3D, mipMapLevels(), colour_fmt, width(), height(), depth());
#	else
		glTexStorage3D(GL_TEXTURE_3D, mipMapLevels(), colour_fmt, width(), height(), depth());
#	endif
	}

	void Texture3D::updateImpl(const TextureResource& data)
	{
		ImageFormat img_fmt = toImageFormat(data.Format != SurfaceFormat::Unknown ? data.Format : format());
		GLsizei w = (GLsizei)data.Width;
		GLsizei h = (GLsizei)data.Height;
		GLsizei d = (GLsizei)data.Depth;
		GLsizei mip = (GLsizei)data.MipMap;

#	if defined(VCL_GL_ARB_direct_state_access)
		glTextureSubImage3D(_glId, 0, 0, 0, 0, w, h, d, img_fmt.Format, img_fmt.Type, data.data());
#	elif defined(VCL_GL_EXT_direct_state_access)
		glTextureSubImage3DEXT(_glId, GL_TEXTURE_3D, 0, 0, 0, 0, w, h, d, img_fmt.Format, img_fmt.Type, data.data());
#	else
		glTexSubImage3D(GL_TEXTURE_3D, mip, 0, 0, 0, w, h, d, img_fmt.Format, img_fmt.Type, data.data());
#	endif
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
