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

#ifdef VCL_OPENGL_SUPPORT

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	TextureCubeArray::TextureCubeArray
	(
		const TextureCubeDescription& desc,
		const TextureResource* init_data /* = nullptr */
	)
	: Texture()
	{
		initializeView
		(
			TextureType::TextureCubeArray, desc.Format,
			0, desc.MipLevels,
			0, desc.ArraySize,
			desc.Width, desc.Height, 1
		);
		initialise(init_data);
	}

	TextureCubeArray::~TextureCubeArray()
	{
		// Delete the texture
		glDeleteTextures(1, &_glId);
	}

	void TextureCubeArray::fill(SurfaceFormat fmt, const void* data)
	{
		ImageFormat gl_fmt = toImageFormat(fmt);
		
		glTextureSubImage3D(_glId, 0, 0, 0, 0, width(), height(), 6 * layers(), gl_fmt.Format, gl_fmt.Type, data);
	}

	void TextureCubeArray::fill(SurfaceFormat fmt, int mip_level, const void* data)
	{
	}

	void TextureCubeArray::read(size_t size, void* data) const
	{
	}

	void TextureCubeArray::initialise(const TextureResource* init_data /* = nullptr */)
	{
		GLenum colour_fmt = toSurfaceFormat(format());
		ImageFormat img_fmt = toImageFormat(format());

		glCreateTextures(GL_TEXTURE_CUBE_MAP_ARRAY, 1, &_glId);
		glTextureStorage3D(_glId, 1, colour_fmt, width(), height(), 6 * layers());
		
		if (init_data)
		{
			glTextureSubImage3D(_glId, 0, 0, 0, 0, init_data->Width, init_data->Height, init_data->Layers, img_fmt.Format, img_fmt.Type, init_data->Data);
		}
		
		// Configure texture
		glTextureParameteri(_glId, GL_TEXTURE_BASE_LEVEL, firstMipMapLevel());
		glTextureParameteri(_glId, GL_TEXTURE_MAX_LEVEL, firstMipMapLevel() + mipMapLevels() - 1);
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
