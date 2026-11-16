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
#include <vcl/config/eigen.h>
#include <vcl/config/direct3d12.h>

// C++ standard library
#include <array>
#include <iostream>

//#include <vcl/graphics/core/enums.h>
//#include <vcl/graphics/resource/depthstencilview.h>
//#include <vcl/graphics/resource/rendertargetview.h>
//#include <vcl/graphics/resource/shaderresourceview.h>

// VCL
//#include <vcl/graphics/d3d12/type_traits.h>
#include <vcl/graphics/surfaceformat.h>

// Support macros
#ifndef VCL_DIRECT3D_SAFE_CALL
#	ifdef VCL_DEBUG
#		define VCL_DIRECT3D_SAFE_CALL(call)                                                     \
			do                                                                                   \
			{                                                                                    \
				const HRESULT vcl_d3d_hr = (call);                                               \
				if (FAILED(vcl_d3d_hr))                                                          \
				{                                                                                \
					std::cout << "D3D Error\tFile: " << __FILE__ << ", " << __LINE__             \
							  << ": HRESULT 0x" << std::hex << static_cast<uint32_t>(vcl_d3d_hr) \
							  << std::dec << std::endl;                                          \
					VCL_DEBUG_BREAK;                                                             \
				}                                                                                \
			} while (VCL_EVAL_FALSE)
#	else
#		define VCL_DIRECT3D_SAFE_CALL(call) call
#	endif
#endif

#ifndef VCL_DIRECT3D_SAFE_RELEASE
#	define VCL_DIRECT3D_SAFE_RELEASE(ptr) \
		if (ptr != 0)                      \
		{                                  \
			ptr->Release();                \
			ptr = 0;                       \
		}
#endif

namespace Vcl { namespace Graphics { namespace D3D12 {
	class D3D
	{
	public:
		//static AnyRenderType toRenderType(SurfaceFormat fmt);

	public:
		static DXGI_FORMAT toD3Denum(SurfaceFormat type);
		//static D3D11_RTV_DIMENSION toD3Denum(RenderTargetViewDimension::RenderTargetViewDimension dim);
		//static D3D11_DSV_DIMENSION toD3Denum(DepthStencilViewDimension::DepthStencilViewDimension dim);
		//static D3D11_SRV_DIMENSION toD3Denum(ShaderResourceViewDimension::ShaderResourceViewDimension dim);
		//
		//static D3D11_QUERY toD3Denum(VCL_ENUM(QueryType) query);
		//
		//static D3D11_BLEND    toD3Denum(Blend::Blend blend);
		//static D3D11_BLEND_OP toD3Denum(BlendOp::BlendOp op);
		//static UINT8          toD3Denum(ColourWriteEnable::ColourWriteEnable mask);
		//
		//static D3D11_FILL_MODE toD3Denum(FillMode::FillMode mode);
		//static D3D11_CULL_MODE toD3Denum(CullMode::CullMode mode);
		//
		//static D3D11_DEPTH_WRITE_MASK toD3Denum(DepthWriteMask::DepthWriteMask mask);
		//static D3D11_COMPARISON_FUNC toD3Denum(ComparisonFunction::ComparisonFunction func);
		//static D3D11_STENCIL_OP toD3Denum(StencilOperation::StencilOperation func);
	};

	// Methods returning structures by value have a different signature in the
	// D3D12 headers for MinGW (the result is passed as an out-parameter).
	inline D3D12_CPU_DESCRIPTOR_HANDLE cpuHeapStart(ID3D12DescriptorHeap* heap)
	{
#if defined(_MSC_VER) || !defined(_WIN32)
		return heap->GetCPUDescriptorHandleForHeapStart();
#else
		D3D12_CPU_DESCRIPTOR_HANDLE handle;
		return *heap->GetCPUDescriptorHandleForHeapStart(&handle);
#endif
	}

	inline D3D12_GPU_DESCRIPTOR_HANDLE gpuHeapStart(ID3D12DescriptorHeap* heap)
	{
#if defined(_MSC_VER) || !defined(_WIN32)
		return heap->GetGPUDescriptorHandleForHeapStart();
#else
		D3D12_GPU_DESCRIPTOR_HANDLE handle;
		return *heap->GetGPUDescriptorHandleForHeapStart(&handle);
#endif
	}

	inline D3D12_RESOURCE_DESC resourceDesc(ID3D12Resource* resource)
	{
#if defined(_MSC_VER) || !defined(_WIN32)
		return resource->GetDesc();
#else
		D3D12_RESOURCE_DESC desc;
		return *resource->GetDesc(&desc);
#endif
	}
}}}
