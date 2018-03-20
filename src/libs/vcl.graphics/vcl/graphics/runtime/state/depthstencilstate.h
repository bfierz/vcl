/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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

// VCL
#include <vcl/core/flags.h>

namespace Vcl { namespace Graphics
{
	VCL_DECLARE_FLAGS(ClearFlag, Depth, Stencil);

	enum class DepthWriteMask
	{
		Zero = 0,
		All = 1
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

	enum class StencilOperation
	{
		//! Keep the existing stencil data.
		Keep = 1,
		//! Set the stencil data to 0.
		Zero = 2,
		//! Set the stencil data to the reference value set
		Replace = 3,
		//! Increment the stencil value by 1, and clamp the result.
		IncreaseSaturate = 4,
		//! Decrement the stencil value by 1, and clamp the result.
		DecreaseSaturate = 5,
		//! Invert the stencil data.
		Invert = 6,
		//! Increment the stencil value by 1, and wrap the result if necessary.
		IncreaseWrap = 7,
		//! Decrement the stencil value by 1, and wrap the result if necessary.
		DecreaseWrap = 8
	};

	struct DepthStencilOperationDescription
	{
		StencilOperation   StencilFailOp{ StencilOperation::Keep };
		StencilOperation   StencilDepthFailOp{ StencilOperation::Keep };
		StencilOperation   StencilPassOp{ StencilOperation::Keep };
		ComparisonFunction StencilFunc{ ComparisonFunction::Always };
	};

	struct DepthStencilDescription
	{
		bool                             DepthEnable{ true };
		DepthWriteMask			         DepthWriteMask{ DepthWriteMask::All };
		ComparisonFunction				 DepthFunc{ ComparisonFunction::Less };
		bool                             StencilEnable{ false };
		uint8_t                          StencilReadMask{ 0xff };
		uint8_t                          StencilWriteMask{ 0xff };
		DepthStencilOperationDescription FrontFace;
		DepthStencilOperationDescription BackFace;
	};
}}