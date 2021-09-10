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

// VCL configuration
#include <vcl/config/global.h>

// C++ Standard Library
#include <algorithm>

// Include the relevant parts from the library
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/graphics/runtime/opengl/graphicsengine.h>

// Google test
#include <gtest/gtest.h>

TEST(OpenGL, EngineFrameCounter)
{
	using namespace Vcl::Graphics::Runtime::OpenGL;

	GraphicsEngine engine;

	engine.beginFrame();
	EXPECT_EQ(engine.currentFrame() % 3, 0);
	engine.endFrame();

	engine.beginFrame();
	EXPECT_EQ(engine.currentFrame() % 3, 1);
	engine.endFrame();

	engine.beginFrame();
	EXPECT_EQ(engine.currentFrame() % 3, 2);
	engine.endFrame();

	engine.beginFrame();
	EXPECT_EQ(engine.currentFrame() % 3, 0);
	engine.endFrame();
}

TEST(OpenGL, EngineCommands)
{
	using namespace Vcl::Graphics::Runtime::OpenGL;

	GraphicsEngine engine;

	bool called = false;

	engine.beginFrame();
	engine.enqueueCommand([&called]() { called = true; });
	engine.endFrame();

	engine.beginFrame();
	engine.endFrame();

	EXPECT_TRUE(called);
}

TEST(OpenGL, EngineRenderTargetUsage)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;
	using namespace Vcl::Core;

	// Engine executing the rendering commands
	Runtime::OpenGL::GraphicsEngine engine;

	// Create the rendertarget
	Texture2DDescription rt_desc;
	rt_desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	rt_desc.ArraySize = 1;
	rt_desc.Width = 32;
	rt_desc.Height = 32;
	rt_desc.MipLevels = 1;
	auto rt_0 = engine.createResource(rt_desc);
	auto rt_1 = engine.createResource(rt_desc);
	auto rt_2 = engine.createResource(rt_desc);

	Texture2DDescription depth_rt_desc;
	depth_rt_desc.Format = SurfaceFormat::D32_FLOAT;
	depth_rt_desc.ArraySize = 1;
	depth_rt_desc.Width = 32;
	depth_rt_desc.Height = 32;
	depth_rt_desc.MipLevels = 1;
	auto depth_rt = engine.createResource(depth_rt_desc);

	// Readback buffers
	std::vector<uint32_t> b_rt_0(32 * 32, 0);
	std::vector<uint32_t> b_rt_1(32 * 32, 0);
	std::vector<uint32_t> b_rt_2(32 * 32, 0);

	engine.beginFrame();
	engine.setRenderTargets({ rt_0 }, depth_rt);
	engine.clear(0, Eigen::Vector4f::Constant(0.25f).eval());
	engine.enqueueReadback(*rt_0, [&b_rt_0](stdext::span<uint8_t> view) { memcpy(b_rt_0.data(), view.data(), view.size()); });
	engine.endFrame();

	engine.beginFrame();
	engine.setRenderTargets({ rt_1 }, depth_rt);
	engine.clear(0, Eigen::Vector4f::Constant(0.5f).eval());
	engine.enqueueReadback(*rt_1, [&b_rt_1](stdext::span<uint8_t> view) { memcpy(b_rt_1.data(), view.data(), view.size()); });
	engine.endFrame();

	engine.beginFrame();
	engine.setRenderTargets({ rt_2 }, depth_rt);
	engine.clear(0, Eigen::Vector4f::Constant(1.0f).eval());
	engine.enqueueReadback(*rt_2, [&b_rt_2](stdext::span<uint8_t> view) { memcpy(b_rt_2.data(), view.data(), view.size()); });
	engine.endFrame();

	// Run three more frames in order to execute the read-back requests
	engine.beginFrame();
	engine.setRenderTargets({ rt_0 }, depth_rt);
	engine.clear(0, Eigen::Vector4f::Constant(1.0f).eval());
	engine.endFrame();
	engine.beginFrame();
	engine.setRenderTargets({ rt_1 }, depth_rt);
	engine.clear(0, Eigen::Vector4f::Constant(0.5f).eval());
	engine.endFrame();
	engine.beginFrame();
	engine.setRenderTargets({ rt_2 }, depth_rt);
	engine.clear(0, Eigen::Vector4f::Constant(0.25f).eval());
	engine.endFrame();

	constexpr uint32_t quarter_0 = 0x3f3f3f3f;
	constexpr uint32_t quarter_1 = 0x40404040;
	constexpr uint32_t half_0 = 0x7f7f7f7f;
	constexpr uint32_t half_1 = 0x80808080;
	constexpr uint32_t one = 0xffffffff;

	// Check read values from each frame
	EXPECT_TRUE(std::all_of(b_rt_0.cbegin(), b_rt_0.cend(), [=](uint32_t v) { return v == quarter_0 || v == quarter_1; }));
	EXPECT_TRUE(std::all_of(b_rt_1.cbegin(), b_rt_1.cend(), [=](uint32_t v) { return v == half_0 || v == half_1; }));
	EXPECT_TRUE(std::all_of(b_rt_2.cbegin(), b_rt_2.cend(), [=](uint32_t v) { return v == 0xffffffff; }));

	// Check the values of each frame
	std::vector<uint32_t> d_rt_0(32 * 32, 0);
	std::vector<uint32_t> d_rt_1(32 * 32, 0);
	std::vector<uint32_t> d_rt_2(32 * 32, 0);
	static_pointer_cast<Runtime::OpenGL::Texture2D>(rt_0)->read(d_rt_0.size() * sizeof(uint32_t), d_rt_0.data());
	static_pointer_cast<Runtime::OpenGL::Texture2D>(rt_1)->read(d_rt_1.size() * sizeof(uint32_t), d_rt_1.data());
	static_pointer_cast<Runtime::OpenGL::Texture2D>(rt_2)->read(d_rt_2.size() * sizeof(uint32_t), d_rt_2.data());

	EXPECT_TRUE(std::all_of(d_rt_0.cbegin(), d_rt_0.cend(), [=](uint32_t v) { return v == 0xffffffff; }));
	EXPECT_TRUE(std::all_of(d_rt_1.cbegin(), d_rt_1.cend(), [=](uint32_t v) { return v == half_0 || v == half_1; }));
	EXPECT_TRUE(std::all_of(d_rt_2.cbegin(), d_rt_2.cend(), [=](uint32_t v) { return v == quarter_0 || v == quarter_1; }));
}

TEST(OpenGL, ConstantBufferUsage)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	// Engine executing the rendering commands
	Runtime::OpenGL::GraphicsEngine engine;

	struct ShaderConstants
	{
		float value;
	};

	// Base address of constants. Must be the same at the start of the next cycle
	void* base_address{ nullptr };

	engine.beginFrame();
	{
		auto memory = engine.requestPerFrameConstantBuffer<ShaderConstants>();
		{
			auto memory2 = engine.requestPerFrameConstantBuffer<ShaderConstants>();
		}
		base_address = memory.data();

		engine.setConstantBuffer(0, std::move(memory));
	}
	engine.endFrame();
	glFinish();

	engine.beginFrame();
	{
		auto memory = engine.requestPerFrameConstantBuffer<ShaderConstants>();
		{
			auto memory2 = engine.requestPerFrameConstantBuffer<ShaderConstants>();
		}
		engine.setConstantBuffer(0, std::move(memory));
	}
	engine.endFrame();
	glFinish();

	engine.beginFrame();
	{
		auto memory = engine.requestPerFrameConstantBuffer<ShaderConstants>();
		{
			auto memory2 = engine.requestPerFrameConstantBuffer<ShaderConstants>();
		}
		engine.setConstantBuffer(0, std::move(memory));
	}
	engine.endFrame();
	glFinish();

	engine.beginFrame();
	{
		auto memory = engine.requestPerFrameConstantBuffer<ShaderConstants>();
		{
			auto memory2 = engine.requestPerFrameConstantBuffer<ShaderConstants>();
		}
		const void* new_base_address = memory.data();

		engine.setConstantBuffer(0, std::move(memory));

		EXPECT_EQ(base_address, new_base_address);
	}
	engine.endFrame();
	glFinish();
}
