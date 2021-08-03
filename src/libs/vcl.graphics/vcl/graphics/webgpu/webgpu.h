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
#include <array>

// VCL
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace WebGPU {
	WGPUTextureFormat toWebGPUEnum(SurfaceFormat type);
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
}}}
