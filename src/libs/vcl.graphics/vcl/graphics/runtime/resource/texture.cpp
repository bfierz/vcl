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
#include <vcl/graphics/runtime/resource/texture.h>

// C++ standard library
#include <algorithm>

// VCL library
#include <vcl/core/contract.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Runtime {
	void Texture::initializeTexture(
		TextureType t,
		const TextureBaseDescription& desc,
		int nrLvls,
		int nrLayers,
		int width,
		int height,
		int depth)
	{
		VclRequire(std::find(desc.ViewFormats.begin(), desc.ViewFormats.end(), SurfaceFormat::Unknown) == desc.ViewFormats.end(), "View formats are valid.");

		_type = t;
		_format = desc.Format;
		_usage = desc.Usage;
		_viewFormats = desc.ViewFormats;
		_nrLevels = nrLvls;
		_nrLayers = nrLayers;
		_width = width;
		_height = height;
		_depth = depth;

		_sizeInBytes = width * height * depth * Vcl::Graphics::sizeInBytes(desc.Format);
	}

	bool Texture::supportsViewFormat(SurfaceFormat format) const
	{
		if (format == SurfaceFormat::Unknown || format == _format)
			return true;

		return std::find(_viewFormats.begin(), _viewFormats.end(), format) != _viewFormats.end();
	}
}}}
