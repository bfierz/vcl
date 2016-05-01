/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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

// C++ standard library
#include <memory>

// GSL
#include <vcl/core/3rdparty/gsl/gsl.h>

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/graphics/runtime/resource/texture.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	/*!
	 *	\brief Holds a several of the same of textures for parallel rendering
	 *
	 *	Each frame that is rendered needs its own version of a texture in order
	 *	to prevent implicit CPU-GPU sync points or data race-conditions.
	 */
	template<int N>
	class DynamicTexture
	{
	public:
		DynamicTexture(std::array<std::unique_ptr<Texture>, N> source)
		{
			_textures = std::move(source);
		}

	public:
		gsl::not_null<Texture*> operator[] (size_t idx) const
		{
			Require(idx < 3, "Index is in range.");

			return _textures[idx].get();
		}

	private:
		std::array<std::unique_ptr<Texture>, N> _textures;
	};
}}}
