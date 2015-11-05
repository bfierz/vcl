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

#ifdef VCL_OPENGL_SUPPORT

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Texture2D::Texture2D
	(
		int w, int h,
		SurfaceFormat fmt,
		const TextureResource* init_data /* = nullptr */
	)
	: Texture()
	{
		initializeView
		(
			TextureType::Texture2D, fmt,
			0, 1,
			0, 1,
			w, h, 1
		);
		initialise(init_data);
	}

	Texture2D::~Texture2D()
	{
		// Delete the texture
		glDeleteTextures(1, &_glId);
	}

	void Texture2D::fill(SurfaceFormat fmt, const void* data)
	{
		ImageFormat gl_fmt = toImageFormat(fmt);

		glTextureSubImage2D(_glId, 0, 0, 0, width(), height(), gl_fmt.Format, gl_fmt.Type, data);
	}

	void Texture2D::fill(SurfaceFormat fmt, int mip_level, const void* data)
	{
		ImageFormat gl_fmt = toImageFormat(fmt);

		glTextureSubImage2D(_glId, mip_level, 0, 0, width(), height(), gl_fmt.Format, gl_fmt.Type, data);
	}

	void Texture2D::read(SurfaceFormat& fmt, void* data) const
	{
		fmt = this->format();
		ImageFormat gl_fmt = toImageFormat(fmt);

		DebugError("Call is buggy");
		glGetTextureImage(_glId, 0, gl_fmt.Format, gl_fmt.Type, 0, data);
	}

	void Texture2D::initialise(const TextureResource* init_data /* = nullptr */)
	{
		GLenum colour_fmt = toSurfaceFormat(format());
		ImageFormat img_fmt = toImageFormat(format());

		glCreateTextures(GL_TEXTURE_2D, 1, &_glId);
		glTextureStorage2D(_glId, 1, colour_fmt, width(), height());

		if (init_data)
		{
			GLsizei w = (GLsizei) init_data->Width;
			GLsizei h = (GLsizei) init_data->Height;
			glTextureSubImage2D(_glId, 0, 0, 0, w, h, img_fmt.Format, img_fmt.Type, init_data->Data);
		}
		
		// Configure texture
		glTextureParameteri(_glId, GL_TEXTURE_BASE_LEVEL, 0);
		glTextureParameteri(_glId, GL_TEXTURE_MAX_LEVEL, 0);
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
