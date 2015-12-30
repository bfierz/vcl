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

// VCL configuration
#include <vcl/config/global.h>

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/graphics/runtime/opengl/state/blendstate.h>

// Google test
#include <gtest/gtest.h>

TEST(OpenGL, ConfigureLogicOp)
{
	using namespace Vcl::Graphics::Runtime::OpenGL;
	using namespace Vcl::Graphics;

	BlendDescription desc;
	desc.LogicOpEnable = true;
	desc.LogicOp = LogicOperation::Clear;

	BlendState state{ desc };
	state.bind();
	EXPECT_TRUE(state.isValid()) << "State is nocht valid";
}

TEST(OpenGL, IndependentBlending)
{
	using namespace Vcl::Graphics::Runtime::OpenGL;
	using namespace Vcl::Graphics;

	if (BlendState::isIndependentBlendingSupported())
	{
		BlendDescription desc;
		desc.IndependentBlendEnable = true;
		desc.RenderTarget[0].BlendEnable = true;
		desc.RenderTarget[0].BlendOp = BlendOperation::Max;
		desc.RenderTarget[1].BlendEnable = true;
		desc.RenderTarget[1].BlendOp = BlendOperation::Min;

		BlendState state{ desc };
		state.bind();
		EXPECT_TRUE(state.isValid()) << "State is nocht valid";
	}
	else
	{
		std::cout << "Independent blending is not supported on this platform!" << std::endl;
	}
}

TEST(OpenGL, AdvancedBlendOp)
{
	using namespace Vcl::Graphics::Runtime::OpenGL;
	using namespace Vcl::Graphics;

	if (BlendState::areAdvancedBlendOperationsSupported())
	{
		BlendDescription desc;
		desc.RenderTarget[0].BlendEnable = true;
		desc.RenderTarget[0].BlendOp = BlendOperation::Screen;

		BlendState state{ desc };
		state.bind();
		EXPECT_TRUE(state.isValid()) << "State is nocht valid";
	}
	else
	{
		std::cout << "Advanced blend operations are not supported on this platform!" << std::endl;
	}
}
