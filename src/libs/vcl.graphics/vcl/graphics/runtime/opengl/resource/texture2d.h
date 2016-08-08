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

// VCL
#include <vcl/graphics/runtime/opengl/resource/texture.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	class Texture2D : public Texture
	{
	public:
		Texture2D(const Texture2DDescription& desc, const TextureResource* init_data = nullptr);
		Texture2D(const Texture2D&);
		virtual ~Texture2D();

	private:
		void initialise(const TextureResource* init_data = nullptr);

	public:
		virtual std::unique_ptr<Runtime::Texture> clone() const override;

	public:
		void copyTo(Buffer& target, size_t dstOffset = 0) const;

	public:
		virtual void fill(SurfaceFormat fmt, const void* data) override;
		virtual void fill(SurfaceFormat fmt, int mip_level, const void* data) override;

		virtual void read(size_t size, void* data) const override;
	};
}}}}
#endif // VCL_OPENGL_SUPPORT
