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
#include <vcl/graphics/surfaceformat.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics {
	int sizeInBytes(SurfaceFormat fmt)
	{
		switch (fmt)
		{
		case SurfaceFormat::R32G32B32A32_FLOAT:
		case SurfaceFormat::R32G32B32A32_UINT:
		case SurfaceFormat::R32G32B32A32_SINT:
			return 16;
		case SurfaceFormat::R16G16B16A16_FLOAT:
		case SurfaceFormat::R16G16B16A16_UNORM:
		case SurfaceFormat::R16G16B16A16_UINT:
		case SurfaceFormat::R16G16B16A16_SNORM:
		case SurfaceFormat::R16G16B16A16_SINT:
			return 8;
		case SurfaceFormat::R32G32B32_FLOAT:
		case SurfaceFormat::R32G32B32_UINT:
		case SurfaceFormat::R32G32B32_SINT:
			return 12;
		case SurfaceFormat::R32G32_FLOAT:
		case SurfaceFormat::R32G32_UINT:
		case SurfaceFormat::R32G32_SINT:
		case SurfaceFormat::D32_FLOAT_S8X24_UINT:
			return 8;
		case SurfaceFormat::R10G10B10A2_UNORM:
		case SurfaceFormat::R10G10B10A2_UINT:
		case SurfaceFormat::R11G11B10_FLOAT:
		case SurfaceFormat::R8G8B8A8_UNORM:
		case SurfaceFormat::R8G8B8A8_UNORM_SRGB:
		case SurfaceFormat::R8G8B8A8_UINT:
		case SurfaceFormat::R8G8B8A8_SNORM:
		case SurfaceFormat::R8G8B8A8_SINT:
		case SurfaceFormat::R16G16_FLOAT:
		case SurfaceFormat::R16G16_UNORM:
		case SurfaceFormat::R16G16_UINT:
		case SurfaceFormat::R16G16_SNORM:
		case SurfaceFormat::R16G16_SINT:
		case SurfaceFormat::D32_FLOAT:
		case SurfaceFormat::R32_FLOAT:
		case SurfaceFormat::R32_UINT:
		case SurfaceFormat::R32_SINT:
		case SurfaceFormat::D24_UNORM_S8_UINT:
			return 4;
		case SurfaceFormat::R8G8B8_UNORM:
			return 3;
		case SurfaceFormat::R8G8_UNORM:
		case SurfaceFormat::R8G8_UINT:
		case SurfaceFormat::R8G8_SNORM:
		case SurfaceFormat::R8G8_SINT:
		case SurfaceFormat::R16_FLOAT:
		case SurfaceFormat::D16_UNORM:
		case SurfaceFormat::R16_UNORM:
		case SurfaceFormat::R16_UINT:
		case SurfaceFormat::R16_SNORM:
		case SurfaceFormat::R16_SINT:
			return 2;
		case SurfaceFormat::R8_UNORM:
		case SurfaceFormat::R8_UINT:
		case SurfaceFormat::R8_SNORM:
		case SurfaceFormat::R8_SINT:
			return 1;
		default:
			VclDebugError("Unknown format");
			return 0;
		}
	}
	bool isDepthFormat(SurfaceFormat fmt)
	{
		switch (fmt)
		{
		case SurfaceFormat::D16_UNORM:
		case SurfaceFormat::D24_UNORM_S8_UINT:
		case SurfaceFormat::D32_FLOAT:
		case SurfaceFormat::D32_FLOAT_S8X24_UINT:
			return true;
		}
		return false;
	}
}}
