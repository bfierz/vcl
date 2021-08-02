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
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/core/contract.h>
#include <vcl/graphics/surfaceformat.h>

// Declaration
namespace Vcl { namespace Graphics {
	enum class ShadowMapType
	{
		None,
		Standard,
		Linear,
		Convolution,
		Exponential,
		Reflective,
		Variance
	};

	class ShadowMap
	{
	protected:
		ShadowMap(ShadowMapType type, unsigned int width, unsigned int height, unsigned int count = 1);

	public:
		virtual ~ShadowMap() = default;

	public:
		//! \returns the type of the shadow map
		ShadowMapType type() const;

		//! \returns the width of the shadow map
		unsigned int width() const;

		//! \returns the height of the shadow map
		unsigned int height() const;

	private:
		//! Type of the shadow map
		ShadowMapType _type;

		//! Width of the shadow map
		unsigned int _width;

		//! Height of the shadow map
		unsigned int _height;

		//! Number of layers in this map
		unsigned int _count;

		//! Description of the single layers
		std::array<SurfaceFormat, 8> _layers;
	};
}}
