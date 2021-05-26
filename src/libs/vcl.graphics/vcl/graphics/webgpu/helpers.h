/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
#include <vcl/config/webgpu.h>

// C++ standard library
#include <tuple>
#include <vector>

// VCL
#include <vcl/core/span.h>
#include <vcl/graphics/webgpu/webgpu.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace WebGPU
{
	auto createTextureFromData(WGPUDevice device, SurfaceFormat type, uint32_t width, uint32_t height, stdext::span<const uint8_t> data)
		-> std::tuple<WGPUTexture, WGPUTextureView>
	{
		std::tuple<WGPUTexture, WGPUTextureView> texture_data;

		WGPUTextureDescriptor tex_desc = {};
		tex_desc.dimension = WGPUTextureDimension_2D;
		tex_desc.size.width = width;
		tex_desc.size.height = height;
		tex_desc.size.depthOrArrayLayers = 1;
		tex_desc.sampleCount = 1;
		tex_desc.format = toWebGPUEnum(type);
		tex_desc.mipLevelCount = 1;
		tex_desc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_Sampled;
		std::get<0>(texture_data) = wgpuDeviceCreateTexture(device, &tex_desc);

		WGPUTextureViewDescriptor tex_view_desc = {};
		tex_view_desc.format = WGPUTextureFormat_RGBA8Unorm;
		tex_view_desc.dimension = WGPUTextureViewDimension_2D;
		tex_view_desc.baseMipLevel = 0;
		tex_view_desc.mipLevelCount = 1;
		tex_view_desc.baseArrayLayer = 0;
		tex_view_desc.arrayLayerCount = 1;
		tex_view_desc.aspect = WGPUTextureAspect_All;
		std::get<1>(texture_data) = wgpuTextureCreateView(std::get<0>(texture_data), &tex_view_desc);

		// Upload texture data
		{
			uint32_t pp_size = sizeInBytes(type);

			WGPUTextureCopyView dst_view = {};
			dst_view.texture = std::get<0>(texture_data);
			dst_view.mipLevel = 0;
			dst_view.origin = { 0, 0, 0 };
			dst_view.aspect = WGPUTextureAspect_All;
			WGPUTextureDataLayout layout = {};
			layout.offset = 0;
			layout.bytesPerRow = width * pp_size;
			layout.rowsPerImage = height;
			WGPUExtent3D size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
			wgpuQueueWriteTexture(wgpuDeviceGetQueue(device), &dst_view, data.data(), data.size(), &layout, &size);
		}

		return texture_data;
	}
}}}
