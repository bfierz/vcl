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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

#ifdef VCL_OPENGL_SUPPORT
// C++ standard library
#include <memory>

// VCL
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/resource/resource.h>
#include <vcl/graphics/runtime/resource/texture.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	struct ImageFormat
	{
		GLenum Format;
		GLenum Type;
	};

	class Texture : public Runtime::Texture, public Resource
	{
	protected:
		Texture() = default;
		Texture(const Texture&);

	public:
		virtual ~Texture() = default;
		
	public:
		static GLenum toSurfaceFormat(SurfaceFormat type);
		static ImageFormat toImageFormat(SurfaceFormat fmt);		

	public:
		void copyTo(Buffer& target, size_t dstOffset = 0) const {};

	public:
		virtual void fill(SurfaceFormat fmt, const void* data) = 0;
		virtual void fill(SurfaceFormat fmt, int mip_level, const void* data) = 0;

		virtual void read(size_t size, void* data) const = 0;
	};
}}}}
#endif // VCL_OPENGL_SUPPORT
