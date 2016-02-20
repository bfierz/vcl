/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#include <vcl/config/opengl.h>

// C++ standard library
#include <vector>

// VCL
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/graphicsengine.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	class Fence final
	{
	public:
		Fence() = default;
		Fence(GLsync sync);
		Fence(Fence&& rhs);
		~Fence();

		Fence& operator=(Fence&& rhs);

	public:
		operator GLsync() const { return _sync; }

	public:
		bool isValid() const { return _sync; }

	private:
		GLsync _sync{ nullptr };
	};

	class Frame final
	{
	public:
		Frame();

	public:
		void mapConstantBuffer();
		void unmapConstantBuffer();

	public:
		const Fence* fence() const { return &_fence; }
		void setFence(Fence fence) { _fence = std::move(fence); }

		ref_ptr<Buffer> constantBuffer() const { return _constantBuffer; }
		void* mappedConstantBuffer() const { return _mappedConstantBuffer; }

	private:
		//! Fence guarding the start of a frame
		Fence _fence;

		//! Buffer used for shader constant data
		owner_ptr<Buffer> _constantBuffer;

		/// Pointer to mapped constant buffer
		void* _mappedConstantBuffer{ nullptr };
	};

	class GraphicsEngine final : public Runtime::GraphicsEngine
	{
	public:
		GraphicsEngine();

	public:
		void beginFrame() override;
		void endFrame() override;
		BufferView requestPerFrameConstantBuffer(size_t size) override;

		void setPipelineState(ref_ptr<Runtime::PipelineState> state) override;

	private:
		//! Number of parallel frames
		const int _numConcurrentFrames{ 3 };

		//! Per frame configuration data
		std::vector<Frame> _frames;

		//! Alignment for uniform buffers
		size_t _cbufferAlignment{ 64 };

		//! Current offset into the constant buffer
		size_t _cbufferOffset{ 0 };
	};
}}}}
