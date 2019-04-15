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

// VCL configuration
#include <vcl/config/global.h>

// Include the relevant parts from the library
#include <vcl/graphics/runtime/opengl/resource/texture1d.h>
#include <vcl/graphics/runtime/opengl/resource/texture1darray.h>
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/graphics/runtime/opengl/resource/texture2darray.h>
#include <vcl/graphics/runtime/opengl/resource/texture3d.h>
#include <vcl/graphics/runtime/opengl/resource/texturecube.h>
#include <vcl/graphics/runtime/opengl/resource/texturecubearray.h>

// Google test
#include <gtest/gtest.h>

std::vector<unsigned char> createTestPattern(int width, int height, int depth)
{
	std::array<unsigned char, 12> pattern =
	{
		0xff, 0x00, 0x00, 0xff,
		0x00, 0xff, 0x00, 0xff,
		0x00, 0x00, 0xff, 0xff,
	};

	std::vector<unsigned char> image;
	image.reserve(width*height*depth*4);
	for (int d = 0; d < depth; d++)
	for (int h = 0; h < height; h++)
	for (int w = 0; w < width*4; w++)
	{
		image.push_back(pattern[4*(h%3) + w%4]);
	}
	return image;
}

void verifySize(const Vcl::Graphics::Runtime::OpenGL::Texture& tex, int exp_w, int exp_h, int exp_d)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	GLenum tex_type = tex.toTextureType(tex.type());

	int w, h, d;
	int miplevel = 0;
#if defined(VCL_GL_ARB_direct_state_access)
	glGetTextureLevelParameteriv(tex.id(), miplevel, GL_TEXTURE_WIDTH, &w);
	glGetTextureLevelParameteriv(tex.id(), miplevel, GL_TEXTURE_HEIGHT, &h);
	glGetTextureLevelParameteriv(tex.id(), miplevel, GL_TEXTURE_DEPTH, &d);
#elif defined(VCL_GL_EXT_direct_state_access)
	if (tex_type != GL_TEXTURE_CUBE_MAP)
	{
		glGetTextureLevelParameterivEXT(tex.id(), tex_type, miplevel, GL_TEXTURE_WIDTH, &w);
		glGetTextureLevelParameterivEXT(tex.id(), tex_type, miplevel, GL_TEXTURE_HEIGHT, &h);
		glGetTextureLevelParameterivEXT(tex.id(), tex_type, miplevel, GL_TEXTURE_DEPTH, &d);
	}
	else
	{
		const GLenum faces[] =
		{
			GL_TEXTURE_CUBE_MAP_POSITIVE_X,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
			GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
			GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
		};
		for (const auto face : faces)
		{
			glGetTextureLevelParameterivEXT(tex.id(), face, miplevel, GL_TEXTURE_WIDTH, &w);
			glGetTextureLevelParameterivEXT(tex.id(), face, miplevel, GL_TEXTURE_HEIGHT, &h);
			glGetTextureLevelParameterivEXT(tex.id(), face, miplevel, GL_TEXTURE_DEPTH, &d);
			EXPECT_EQ(exp_w, w);
			EXPECT_EQ(exp_h, h);
			EXPECT_EQ(exp_d, d);
		}
	}
#else
	Runtime::OpenGL::TextureBindPoint bp(tex_type, tex.id());
	if (tex_type != GL_TEXTURE_CUBE_MAP)
	{
		glGetTexLevelParameteriv(tex_type, miplevel, GL_TEXTURE_WIDTH, &w);
		glGetTexLevelParameteriv(tex_type, miplevel, GL_TEXTURE_HEIGHT, &h);
		glGetTexLevelParameteriv(tex_type, miplevel, GL_TEXTURE_DEPTH, &d);
	}
	else
	{
		const GLenum faces[] =
		{
			GL_TEXTURE_CUBE_MAP_POSITIVE_X,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
			GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
			GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
		};
		for (const auto face : faces)
		{
			glGetTexLevelParameteriv(face, miplevel, GL_TEXTURE_WIDTH, &w);
			glGetTexLevelParameteriv(face, miplevel, GL_TEXTURE_HEIGHT, &h);
			glGetTexLevelParameteriv(face, miplevel, GL_TEXTURE_DEPTH, &d);
			EXPECT_EQ(exp_w, w);
			EXPECT_EQ(exp_h, h);
			EXPECT_EQ(exp_d, d);
		}
	}
#endif
	EXPECT_EQ(exp_w, w) << "Texture has wrong width. Expected " << exp_w << ", got " << w;
	EXPECT_EQ(exp_h, h) << "Texture has wrong height/layer count. Expected " << exp_h << ", got " << h;
	EXPECT_EQ(exp_d, d) << "Texture has wrong depth/layer count. Expected " << exp_d << ", got " << d;
}

TEST(OpenGL, InitEmptyTexture1D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.ArraySize = 1;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;
	Runtime::OpenGL::Texture1D tex{ desc1d };
	
	verifySize(tex, 32, 1, 1);
}

TEST(OpenGL, InitTexture1D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.ArraySize = 1;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;

	const auto test_image = createTestPattern(32, 1, 1);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	Runtime::OpenGL::Texture1D tex{ desc1d, &res };

	std::vector<unsigned char> image(32 * 1 * 4);
	tex.read(image.size(), image.data());
	EXPECT_EQ(image, test_image);
}


TEST(OpenGL, InitEmptyTexture1DArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.ArraySize = 2;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;
	Runtime::OpenGL::Texture1DArray tex{ desc1d };

	verifySize(tex, 32, 2, 1);
}

TEST(OpenGL, InitTexture1DArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.ArraySize = 2;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;

	const auto test_image = createTestPattern(32, 2, 1);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Layers = 2;
	res.Width = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	Runtime::OpenGL::Texture1DArray tex{ desc1d, &res };

	std::vector<unsigned char> image(32 * 2 * 4);
	tex.read(image.size(), image.data());
	EXPECT_EQ(image, test_image);
}

TEST(OpenGL, InitEmptyTexture2D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 1;
	desc2d.Width = 32;
	desc2d.Height = 32;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2D tex{ desc2d };

	verifySize(tex, 32, 32, 1);
}

TEST(OpenGL, InitTexture2D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 1;
	desc2d.Width = 32;
	desc2d.Height = 32;
	desc2d.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 1);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = 32;
	res.Height = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	Runtime::OpenGL::Texture2D tex{ desc2d, &res };

	std::vector<unsigned char> image(32 * 32 * 4);
	tex.read(image.size(), image.data());
	EXPECT_EQ(image, test_image);
}

TEST(OpenGL, InitEmptyTexture2DArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 2;
	desc2d.Width = 32;
	desc2d.Height = 32;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2DArray tex{ desc2d };

	verifySize(tex, 32, 32, 2);
}

TEST(OpenGL, InitTexture2DArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 2;
	desc2d.Width = 32;
	desc2d.Height = 32;
	desc2d.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 2);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Layers = 2;
	res.Width = 32;
	res.Height = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	Runtime::OpenGL::Texture2DArray tex{ desc2d, &res };

	std::vector<unsigned char> image(32 * 32 * 2 * 4);
	tex.read(image.size(), image.data());
	EXPECT_EQ(image, test_image);
}

TEST(OpenGL, InitEmptyTexture3D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture3DDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.Width = 32;
	desc.Height = 32;
	desc.Depth = 2;
	desc.MipLevels = 1;
	Runtime::OpenGL::Texture3D tex{ desc };

	verifySize(tex, 32, 32, 2);
}

TEST(OpenGL, InitTexture3D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture3DDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.Width = 32;
	desc.Height = 32;
	desc.Depth = 2;
	desc.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 2);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = 32;
	res.Height = 32;
	res.Depth = 2;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	Runtime::OpenGL::Texture3D tex{ desc, &res };

	std::vector<unsigned char> image(32 * 32 * 2 * 4);
	tex.read(image.size(), image.data());
	EXPECT_EQ(image, test_image);
}

TEST(OpenGL, InitEmptyTextureCube)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	TextureCubeDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.Width = 32;
	desc.Height = 32;
	desc.MipLevels = 1;
	Runtime::OpenGL::TextureCube tex{ desc };

	verifySize(tex, 32, 32, 1);
}

TEST(OpenGL, InitTextureCube)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	TextureCubeDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.Width = 32;
	desc.Height = 32;
	desc.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 6);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = 32;
	res.Height = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	Runtime::OpenGL::TextureCube tex{ desc, &res };

	std::vector<unsigned char> image(32 * 32 * 6 * 4);
	tex.read(image.size(), image.data());
	EXPECT_EQ(image, test_image);
}

TEST(OpenGL, InitEmptyTextureCubeArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	TextureCubeDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.ArraySize = 2;
	desc.Width = 32;
	desc.Height = 32;
	desc.MipLevels = 1;
	Runtime::OpenGL::TextureCubeArray tex{ desc };

	verifySize(tex, 32, 32, 12);
}

TEST(OpenGL, InitTextureCubeArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	TextureCubeDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.ArraySize = 2;
	desc.Width = 32;
	desc.Height = 32;
	desc.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 12);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Layers = 2;
	res.Width = 32;
	res.Height = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	Runtime::OpenGL::TextureCubeArray tex{ desc, &res };

	std::vector<unsigned char> image(32 * 32 * 12 * 4);
	tex.read(image.size(), image.data());
	EXPECT_EQ(image, test_image);
}
