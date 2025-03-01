/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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

namespace Vcl { namespace Graphics { namespace Runtime {
	enum class FillModeMethod
	{
		Wireframe = 2,
		Solid = 3
	};

	enum class CullModeMethod
	{
		None = 1,
		Front = 2,
		Back = 3
	};

	struct RasterizerDescription
	{
		FillModeMethod FillMode{ FillModeMethod::Solid };
		CullModeMethod CullMode{ CullModeMethod::Back };
		bool FrontCounterClockwise{ true };
		int DepthBias{ 0 };
		float SlopeScaledDepthBias{ 0.0f };
		float DepthBiasClamp{ 0.0f };
		bool DepthClipEnable{ true };
		bool ScissorEnable{ false };
		bool MultisampleEnable{ false };
		bool AntialiasedLineEnable{ false };
	};
}}}
