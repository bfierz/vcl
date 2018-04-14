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

// VCL
#include <vcl/core/flags.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	VCL_DECLARE_FLAGS(ColourWriteEnable, Red, Green, Blue, Alpha)

	enum class Blend
	{
		Zero = 1,
		One = 2,
		SrcColour = 3,
		InvSrcColour = 4,
		SrcAlpha = 5,
		InvSrcAlpha = 6,
		DestAlpha = 7,
		InvDestAlpha = 8,
		DestColour = 9,
		InvDestColour = 10,
		SrcAlphaSat = 11,
		BlendFactor = 14,
		InvBlendFactor = 15,
		Src1Colour = 16,
		InvSrc1Colour = 17,
		Src1Alpha = 18,
		InvSrc1Alpha = 19
	};

	enum class BlendOperation
	{
		Add = 1,
		Subtract = 2,
		RevSubtract = 3,
		Min = 4,
		Max = 5,

		// Advanced blend functions
		Multiply = 6,
		Screen = 7,
		Overlay = 8,
		Darken = 9,
		Lighten = 10,
		Colordodge = 11,
		Colorburn = 12,
		Hardlight = 13,
		Softlight = 14, 
		Difference = 15,
		Exclusion = 16,

		HslHue = 17,
		HslSaturation = 18,
		HslColor = 19,
		HslLuminosity = 20
	};

	enum class LogicOperation
	{
		Clear = 0,
		Set = (Clear + 1),
		Copy = (Set + 1),
		CopyInverted = (Copy + 1),
		NoOp = (CopyInverted + 1),
		Invert = (NoOp + 1),
		And = (Invert + 1),
		Nand = (And + 1),
		Or = (Nand + 1),
		Nor = (Or + 1),
		Xor = (Nor + 1),
		Equiv = (Xor + 1),
		AndReverse = (Equiv + 1),
		AndInverted = (AndReverse + 1),
		OrReverse = (AndInverted + 1),
		OrInverted = (OrReverse + 1)
	};

	struct RenderTargetBlendDescription
	{
		bool					 BlendEnable{ false };
		Blend					 SrcBlend{ Blend::One };
		Blend					 DestBlend{ Blend::Zero };
		BlendOperation			 BlendOp{ BlendOperation::Add };
		Blend					 SrcBlendAlpha{ Blend::One };
		Blend					 DestBlendAlpha{ Blend::Zero };
		BlendOperation			 BlendOpAlpha{ BlendOperation::Add };
		Flags<ColourWriteEnable> RenderTargetWriteMask{ ColourWriteEnable::Red | ColourWriteEnable::Green | ColourWriteEnable::Blue | ColourWriteEnable::Alpha };
	};

	struct BlendDescription
	{
		bool AlphaToCoverageEnable{ false };
		bool IndependentBlendEnable{ false };

		bool           LogicOpEnable{ false };
		LogicOperation LogicOp{ LogicOperation::NoOp };
		std::array<RenderTargetBlendDescription, 8> RenderTarget;

		std::array<float, 4> ConstantColor;
	};
}}}
