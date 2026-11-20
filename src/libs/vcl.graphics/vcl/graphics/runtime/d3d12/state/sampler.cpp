/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2026 Basil Fierz
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
#include <vcl/graphics/runtime/d3d12/state/sampler.h>

// C++ standard library
#include <algorithm>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 {
	namespace {
		// 'FilterType', 'TextureAddressMode' and 'ComparisonFunction' use the D3D12 encoding
		D3D12_FILTER toD3D12(FilterType filter) { return static_cast<D3D12_FILTER>(filter); }
		D3D12_TEXTURE_ADDRESS_MODE toD3D12(TextureAddressMode mode) { return static_cast<D3D12_TEXTURE_ADDRESS_MODE>(mode); }
		D3D12_COMPARISON_FUNC toD3D12(ComparisonFunction func) { return static_cast<D3D12_COMPARISON_FUNC>(func); }

		//! Anisotropic filters take 1 to 16 samples
		UINT toMaxAnisotropy(unsigned int max_anisotropy)
		{
			return std::min(std::max(max_anisotropy, 1u), static_cast<unsigned int>(D3D12_MAX_MAXANISOTROPY));
		}

		//! Static samplers only provide a fixed set of border colours
		D3D12_STATIC_BORDER_COLOR toD3D12StaticBorderColor(const std::array<float, 4>& colour)
		{
			const std::array<float, 4> opaque_black = { 0, 0, 0, 1 };
			const std::array<float, 4> opaque_white = { 1, 1, 1, 1 };

			if (colour == opaque_black)
				return D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
			if (colour == opaque_white)
				return D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;

			VclRequire((colour == std::array<float, 4>{ 0, 0, 0, 0 }), "Border colour is transparent black, opaque black or opaque white.");
			return D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		}
	}

	D3D12_SAMPLER_DESC toD3D12(const SamplerDescription& desc)
	{
		D3D12_SAMPLER_DESC d3d12_desc = {};
		d3d12_desc.Filter = toD3D12(desc.Filter);
		d3d12_desc.AddressU = toD3D12(desc.AddressU);
		d3d12_desc.AddressV = toD3D12(desc.AddressV);
		d3d12_desc.AddressW = toD3D12(desc.AddressW);
		d3d12_desc.MipLODBias = desc.MipLODBias;
		d3d12_desc.MaxAnisotropy = toMaxAnisotropy(desc.MaxAnisotropy);
		d3d12_desc.ComparisonFunc = toD3D12(desc.ComparisonFunc);
		std::copy(desc.BorderColor.begin(), desc.BorderColor.end(), d3d12_desc.BorderColor);
		d3d12_desc.MinLOD = desc.MinLOD;
		d3d12_desc.MaxLOD = desc.MaxLOD;

		return d3d12_desc;
	}

	D3D12_STATIC_SAMPLER_DESC toD3D12(const StaticSampler& sampler)
	{
		const auto& desc = sampler.Sampler;

		D3D12_STATIC_SAMPLER_DESC d3d12_desc = {};
		d3d12_desc.Filter = toD3D12(desc.Filter);
		d3d12_desc.AddressU = toD3D12(desc.AddressU);
		d3d12_desc.AddressV = toD3D12(desc.AddressV);
		d3d12_desc.AddressW = toD3D12(desc.AddressW);
		d3d12_desc.MipLODBias = desc.MipLODBias;
		d3d12_desc.MaxAnisotropy = toMaxAnisotropy(desc.MaxAnisotropy);
		d3d12_desc.ComparisonFunc = toD3D12(desc.ComparisonFunc);
		d3d12_desc.BorderColor = toD3D12StaticBorderColor(desc.BorderColor);
		d3d12_desc.MinLOD = desc.MinLOD;
		d3d12_desc.MaxLOD = desc.MaxLOD;
		d3d12_desc.ShaderRegister = sampler.Register;
		d3d12_desc.RegisterSpace = 1;
		d3d12_desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		return d3d12_desc;
	}
}}}}
