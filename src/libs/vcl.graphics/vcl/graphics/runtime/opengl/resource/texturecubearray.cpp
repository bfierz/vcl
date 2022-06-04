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
#include <vcl/graphics/runtime/opengl/resource/texturecubearray.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL {
	TextureCubeArray::TextureCubeArray(
		const TextureCubeDescription& desc,
		const TextureResource* init_data)
	{
		initializeView(
			TextureType::TextureCubeArray, desc.Format, desc.Usage,
			0, desc.MipLevels,
			0, desc.ArraySize,
			desc.Width, desc.Height, 1);
		initialise(init_data);
	}

	std::unique_ptr<Runtime::Texture> TextureCubeArray::clone() const
	{
		return std::make_unique<TextureCubeArray>(*this);
	}

	void TextureCubeArray::allocImpl(GLenum colour_fmt)
	{
#if defined(VCL_GL_ARB_direct_state_access)
		glTextureStorage3D(_glId, mipMapLevels(), colour_fmt, width(), height(), 6 * layers());
#elif defined(VCL_GL_EXT_direct_state_access)
		glTextureStorage3DEXT(_glId, GL_TEXTURE_CUBE_MAP_ARRAY, mipMapLevels(), colour_fmt, width(), height(), 6 * layers());
#else
		glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipMapLevels(), colour_fmt, width(), height(), 6 * layers());
#endif
	}

	void TextureCubeArray::updateImpl(const TextureResource& data)
	{
		ImageFormat img_fmt = toImageFormat(data.Format != SurfaceFormat::Unknown ? data.Format : format());
		GLsizei w = (GLsizei)data.Width;
		GLsizei h = (GLsizei)data.Height;
		GLsizei l = (GLsizei)data.Layers;
		GLsizei mip = (GLsizei)data.MipMap;

#if defined(VCL_GL_ARB_direct_state_access)
		glTextureSubImage3D(_glId, mip, 0, 0, 0, w, h, 6 * l, img_fmt.Format, img_fmt.Type, data.data());
#elif defined(VCL_GL_EXT_direct_state_access)
		glTextureSubImage3DEXT(_glId, GL_TEXTURE_CUBE_MAP_ARRAY, mip, 0, 0, 0, w, h, 6 * l, img_fmt.Format, img_fmt.Type, data.data());
#else
		glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mip, 0, 0, 0, w, h, 6 * l, img_fmt.Format, img_fmt.Type, data.data());
#endif
	}
}}}}
