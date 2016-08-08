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
#include <mutex>
#include <vector>
#include <unordered_map>

// VCL
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/state/framebuffer.h>
#include <vcl/graphics/runtime/graphicsengine.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	class Frame;

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
		bool isValid() const { return _sync != nullptr; }

	private:
		GLsync _sync{ nullptr };
	};

	class StagingArea final
	{
	public:
		StagingArea(Frame* frame);

	public:
		void transfer();

	public:
		BufferView copyFrom(const BufferView& view);
		BufferView copyFrom(const TextureView& view);

	private:
		void updateIfNeeded(size_t size);

	private:
		//! Owning frame
		Frame* _frame{ nullptr };

		//! Backend staging buffer
		owner_ptr<Buffer> _stagingBuffer;
		
		//! Host copy of the GPU memory
		std::unique_ptr<char[]> _hostBuffer;

		//! Current offset in the buffer
		size_t _stagingBufferOffset{ 0 };

		//! Staging buffer alignment
		size_t _alignment{ 256 };

	private:
		//! List of read-back requrests
		std::vector<BufferView> _requests;
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

		StagingArea* readBackBuffer() { return &_readBackStage; }

		void setRenderTargets(gsl::span<Runtime::Texture*> colour_targets, Runtime::Texture* depth_target);

		void queueBufferForDeletion(owner_ptr<Buffer> buffer);

	private:
		//! Fence guarding the start of a frame
		Fence _fence;

		//! Buffer used for shader constant data
		owner_ptr<Buffer> _constantBuffer;

		/// Pointer to mapped constant buffer
		void* _mappedConstantBuffer{ nullptr };

	private:
		//! Framebuffer cache
		std::unordered_map<size_t, OpenGL::Framebuffer> _fbos;

		//! Staging area for data read-back
		StagingArea _readBackStage;

		//! Queue buffers for deletion
		std::vector<owner_ptr<Buffer>> _bufferRecycleQueue;
	};

	class GraphicsEngine final : public Runtime::GraphicsEngine
	{
	public:
		GraphicsEngine();

	public:
		void beginFrame() override;
		void endFrame() override;
		
		BufferView requestPerFrameConstantBuffer(size_t size) override;
		ref_ptr<DynamicTexture<3>> allocatePersistentTexture(std::unique_ptr<Runtime::Texture> tex) override;
		void deletePersistentTexture(ref_ptr<DynamicTexture<3>> tex) override;

		void queueReadback(ref_ptr<DynamicTexture<3>> tex) override;

		void enqueueCommand(std::function<void(void)>) override;

		void resetRenderTargets() override;

		void setRenderTargets(gsl::span<ref_ptr<Runtime::Texture>> colour_targets, ref_ptr<Runtime::Texture> depth_target) override;
		void setRenderTargets(gsl::span<ref_ptr<DynamicTexture<3>>> colour_targets, ref_ptr<Runtime::Texture> depth_target) override;
		void setRenderTargets(gsl::span<ref_ptr<Runtime::Texture>> colour_targets, ref_ptr<DynamicTexture<3>> depth_target) override;
		void setRenderTargets(gsl::span<ref_ptr<DynamicTexture<3>>> colour_targets, ref_ptr<DynamicTexture<3>> depth_target) override;
		void setConstantBuffer(int idx, BufferView buffer) override;

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

	private: // Command management

		//! Command lock
		std::mutex _cmdMutex;

		//! List of generic commands executed at frame start
		std::vector<std::function<void(void)>> _genericCmds;

	private: // Resource held by the engine

		//! Dynamic textures
		std::vector<owner_ptr<DynamicTexture<3>>> _persistentTextures;

	private: // Tracking state
		Frame* _current_frame{ nullptr };
	};
}}}}
