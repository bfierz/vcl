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

// C++ standard library
#include <array>

namespace Vcl { namespace Graphics { namespace Runtime
{
	enum class Filter
	{
		MinMagMipPoint = 0,
		MinMagPointMipLinear = 0x1,
		MinPointMagLinearMipPoint = 0x4,
		MinPointMagMipLinear = 0x5,
		MinLinearMagMipPoint = 0x10,
		MinLinearMagPointMipLinear = 0x11,
		MinMagLinearMipPoint = 0x14,
		MinMagMipLinear = 0x15,
		Anisotropic = 0x55,
		ComparisonMinMagMipPoint = 0x80,
		ComparisonMinMagPointMipLinear = 0x81,
		ComparisonMinPointMagLinearMipPoint = 0x84,
		ComparisonMinPointMagMipLinear = 0x85,
		ComparisonMinLinearMagMipPoint = 0x90,
		ComparisonMinLinearMagPointMipLinear = 0x91,
		ComparisonMinMagLinearMipPoint = 0x94,
		ComparisonMinMagMipLinear = 0x95,
		ComparisonAnisotropic = 0xd5
	};

	enum class ComparisonFunction
	{
		Never = 1,
		Less = 2,
		Equal = 3,
		LessEqual = 4,
		Greater = 5,
		NotEqual = 6,
		GreaterEqual = 7,
		Always = 8
	};

	enum class TextureAddressMode
	{
		Wrap = 1,
		Mirror = 2,
		Clamp = 3,
		Border = 4,
		MirrorOnce = 5
	};

	struct SamplerDescription
	{
		SamplerDescription();

		Filter  		     Filter;
		TextureAddressMode   AddressU;
		TextureAddressMode   AddressV;
		TextureAddressMode   AddressW;
		float			     MipLODBias;
		unsigned int	     MaxAnisotropy;
		ComparisonFunction   ComparisonFunc;
		std::array<float, 4> BorderColor;
		float				 MinLOD;
		float				 MaxLOD;
	};

	class Sampler
	{
	public:
		Sampler(const SamplerDescription& desc);

	public:
		const SamplerDescription& desc() const { return _desc; }
		
	private: // Descriptor
		SamplerDescription _desc;
	};
}}}
