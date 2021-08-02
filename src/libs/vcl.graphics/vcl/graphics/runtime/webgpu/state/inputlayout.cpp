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
#include <vcl/graphics/runtime/webgpu/state/inputlayout.h>

// VCL

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU {
	WGPUVertexFormat toWebGPU(SurfaceFormat format)
	{
		switch (format)
		{
		case SurfaceFormat::R8G8_UINT:          return WGPUVertexFormat_Uint8x2;
		case SurfaceFormat::R8G8B8A8_UINT:      return WGPUVertexFormat_Uint8x4;
		case SurfaceFormat::R8G8_SINT:          return WGPUVertexFormat_Sint8x2;
		case SurfaceFormat::R8G8B8A8_SINT:      return WGPUVertexFormat_Sint8x4;
		case SurfaceFormat::R8G8_UNORM:         return WGPUVertexFormat_Unorm8x2;
		case SurfaceFormat::R8G8B8A8_UNORM:     return WGPUVertexFormat_Unorm8x4;
		case SurfaceFormat::R8G8_SNORM:         return WGPUVertexFormat_Snorm8x2;
		case SurfaceFormat::R8G8B8A8_SNORM:     return WGPUVertexFormat_Snorm8x4;
		case SurfaceFormat::R16G16_UINT:        return WGPUVertexFormat_Uint16x2;
		case SurfaceFormat::R16G16B16A16_UINT:  return WGPUVertexFormat_Uint16x4;
		case SurfaceFormat::R16G16_SINT:        return WGPUVertexFormat_Sint16x2;
		case SurfaceFormat::R16G16B16A16_SINT:  return WGPUVertexFormat_Sint16x4;
		case SurfaceFormat::R16G16_UNORM:       return WGPUVertexFormat_Unorm16x2;
		case SurfaceFormat::R16G16B16A16_UNORM: return WGPUVertexFormat_Unorm16x4;
		case SurfaceFormat::R16G16_SNORM:       return WGPUVertexFormat_Snorm16x2;
		case SurfaceFormat::R16G16B16A16_SNORM: return WGPUVertexFormat_Snorm16x4;
		case SurfaceFormat::R16G16_FLOAT:       return WGPUVertexFormat_Float16x2;
		case SurfaceFormat::R16G16B16A16_FLOAT: return WGPUVertexFormat_Float16x4;
		case SurfaceFormat::R32_FLOAT:          return WGPUVertexFormat_Float32;
		case SurfaceFormat::R32G32_FLOAT:       return WGPUVertexFormat_Float32x2;
		case SurfaceFormat::R32G32B32_FLOAT:    return WGPUVertexFormat_Float32x3;
		case SurfaceFormat::R32G32B32A32_FLOAT: return WGPUVertexFormat_Float32x4;
		case SurfaceFormat::R32_UINT:           return WGPUVertexFormat_Uint32;
		case SurfaceFormat::R32G32_UINT:        return WGPUVertexFormat_Uint32x2;
		case SurfaceFormat::R32G32B32_UINT:     return WGPUVertexFormat_Uint32x3;
		case SurfaceFormat::R32G32B32A32_UINT:  return WGPUVertexFormat_Uint32x4;
		case SurfaceFormat::R32_SINT:           return WGPUVertexFormat_Sint32;
		case SurfaceFormat::R32G32_SINT:        return WGPUVertexFormat_Sint32x2;
		case SurfaceFormat::R32G32B32_SINT:     return WGPUVertexFormat_Sint32x3;
		case SurfaceFormat::R32G32B32A32_SINT:  return WGPUVertexFormat_Sint32x4;
		}

		return WGPUVertexFormat_Force32;
	}

	std::pair<std::vector<WGPUVertexBufferLayoutDescriptor>, std::vector<WGPUVertexAttributeDescriptor>> toWebGPU(const InputLayoutDescription& desc)
	{
		std::vector<WGPUVertexBufferLayoutDescriptor> webgpu_buffer_desc;
		webgpu_buffer_desc.reserve(desc.attributes().size());
		std::vector<WGPUVertexAttributeDescriptor> webgpu_attrib_desc;
		webgpu_attrib_desc.reserve(desc.attributes().size());

		int idx = 0;
		for (const auto& elem : desc.attributes())
		{
			const auto& binding = desc.binding(elem.InputSlot);

			WGPUVertexBufferLayoutDescriptor webgpu_elem;
			webgpu_elem.arrayStride = binding.Stride;
			webgpu_elem.stepMode = binding.InputRate == VertexDataClassification::VertexDataPerObject ? WGPUInputStepMode_Vertex : WGPUInputStepMode_Instance;
			webgpu_elem.attributeCount = elem.NumberLocations;
			webgpu_buffer_desc.emplace_back(webgpu_elem);

			for (int sub_loc = 0; sub_loc < std::max(1, (int)elem.NumberLocations); sub_loc++)
			{
				WGPUVertexAttributeDescriptor webgpu_attribute;
				webgpu_attribute.format = toWebGPU(elem.Format);
				webgpu_attribute.offset = elem.Offset;
				webgpu_attribute.shaderLocation = desc.location(idx) + sub_loc;
				webgpu_attrib_desc.emplace_back(webgpu_attribute);
			}

			idx++;
		}

		return { webgpu_buffer_desc, webgpu_attrib_desc };
	}
}}}}
