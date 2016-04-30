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

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>

// Google test
#include <gtest/gtest.h>

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
	OpenGL::Texture2D tex{ desc2d };

	// Verify the result
	int w, h;
	int miplevel = 0;
	glGetTextureLevelParameteriv(tex.id(), miplevel, GL_TEXTURE_WIDTH, &w);
	glGetTextureLevelParameteriv(tex.id(), miplevel, GL_TEXTURE_HEIGHT, &h);

	EXPECT_EQ(32, w) << "Texture has wrong width.";
	EXPECT_EQ(32, h) << "Texture has wrong height.";
}
