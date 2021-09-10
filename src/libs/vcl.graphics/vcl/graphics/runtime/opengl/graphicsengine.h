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
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/state/framebuffer.h>
#include <vcl/graphics/runtime/graphicsengine.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL {
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
		stdext::span<uint8_t> copyFrom(BufferRange view);
		stdext::span<uint8_t> copyFrom(const TextureView& view);

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
		std::vector<BufferRange> _requests;
	};

	class Frame final
	{
	public:
		Frame();

	public:
		void mapBuffers();
		void unmapBuffers();

	public:
		const Fence* fence() const { return &_fence; }
		void setFence(Fence fence) { _fence = std::move(fence); }

		ref_ptr<Buffer> constantBuffer() const { return _constantBuffer; }
		void* mappedConstantBuffer() const { return _mappedConstantBuffer; }

		ref_ptr<Buffer> linearMemoryBuffer() const { return _linearMemoryBuffer; }
		void* mappedLinearMemoryBuffer() const { return _mappedLinearMemory; }

		void enqueueReadback(const Runtime::Texture& tex, std::function<void(stdext::span<uint8_t>)> callback)
		{
			// Enqueue the copy command
			auto memory_range = _readBackStage.copyFrom(tex);

			// Enqueue the callback
			_readBackCallbacks.emplace_back(memory_range, std::move(callback));
		}

		void executeReadbackRequests()
		{
			// Execute the GPU->CPU memory transfers
			_readBackStage.transfer();

			// Execute the callbacks of the read-back requests
			for (auto& callback : _readBackCallbacks)
			{
				callback.second(callback.first);
			}
			_readBackCallbacks.clear();
		}

		OpenGL::Framebuffer* currentFramebuffer() { return _currentFramebuffer; }
		void setRenderTargets(stdext::span<Runtime::Texture*> colour_targets, Runtime::Texture* depth_target);
		void setRenderTargets(stdext::span<ref_ptr<Runtime::Texture>> colour_targets, ref_ptr<Runtime::Texture> depth_target);

		void queueBufferForDeletion(owner_ptr<Buffer> buffer);

	private:
		//! Fence guarding the start of a frame
		Fence _fence;

		//! Buffer used for shader constant data
		owner_ptr<Buffer> _constantBuffer;

		/// Pointer to mapped constant buffer
		void* _mappedConstantBuffer{ nullptr };

		//! Buffer used for linear memory
		owner_ptr<Buffer> _linearMemoryBuffer;

		/// Pointer to mapped constant buffer
		void* _mappedLinearMemory{ nullptr };

	private:
		//! Framebuffer cache
		std::unordered_map<size_t, OpenGL::Framebuffer> _fbos;

		//! Currently bound framebuffer
		OpenGL::Framebuffer* _currentFramebuffer{ nullptr };

		//! Staging area for data read-back
		StagingArea _readBackStage;

		//! Read-back requests
		std::vector<std::pair<stdext::span<uint8_t>, std::function<void(stdext::span<uint8_t>)>>> _readBackCallbacks;

		//! Queue buffers for deletion
		std::vector<owner_ptr<Buffer>> _bufferRecycleQueue;
	};

	class GraphicsEngine final : public Runtime::GraphicsEngine
	{
	public:
		GraphicsEngine();

	public:
		owner_ptr<Runtime::Texture> createResource(const Texture2DDescription& desc) override;
		owner_ptr<Runtime::Buffer> createResource(const BufferDescription& desc) override;

		void beginFrame() override;
		void endFrame() override;

		using Runtime::GraphicsEngine::requestPerFrameConstantBuffer;
		BufferView requestPerFrameConstantBuffer(size_t size) override;
		BufferView requestPerFrameLinearMemory(size_t size) override;

		void enqueueReadback(const Runtime::Texture& tex, std::function<void(stdext::span<uint8_t>)> callback) override;

		void enqueueCommand(std::function<void(void)>) override;

		// Expose method from parent class
		using Runtime::GraphicsEngine::setRenderTargets;

		[[deprecated]] void setRenderTargets(stdext::span<const ref_ptr<Runtime::Texture>> colour_targets, ref_ptr<Runtime::Texture> depth_target) override;

		void setConstantBuffer(int idx, BufferView buffer) override;
		void setIndexBuffer(const Runtime::Buffer& buffer) override;
		void setVertexBuffer(int idx, const Runtime::Buffer& buffer, int offset, int stride) override;
		void setSampler(int idx, const Runtime::Sampler& sampler) override;
		void setSamplers(int idx, stdext::span<const ref_ptr<Runtime::Sampler>> samplers) override;
		void setTexture(int idx, const Runtime::Texture& texture) override;
		void setTextures(int idx, stdext::span<const ref_ptr<Runtime::Texture>> textures) override;

		void setPipelineState(ref_ptr<Runtime::PipelineState> state) override;

		//! Set per draw-call shader parameters
		[[deprecated]] void pushConstants(void* data, size_t size) override;

		void beginRenderPass(const RenderPassDescription& pass_desc) override;
		void endRenderPass() override;
		[[deprecated]] void clear(int idx, const Eigen::Vector4f& colour) override;
		[[deprecated]] void clear(int idx, const Eigen::Vector4i& colour) override;
		[[deprecated]] void clear(int idx, const Eigen::Vector4ui& colour) override;
		[[deprecated]] void clear(float depth, int stencil) override;
		[[deprecated]] void clear(float depth) override;
		[[deprecated]] void clear(int stencil) override;

		void setPrimitiveType(PrimitiveType type, int nr_vertices = -1) override;
		void draw(int count, int first = 0, int instance_count = 1, int base_instance = 0) override;
		void drawIndexed(int count, int first_index = 0, int instance_count = 1, int base_vertex = 0, int base_instance = 0) override;

	private:
		//! Number of parallel frames
		const int _numConcurrentFrames{ 3 };

		//! Per frame configuration data
		std::vector<Frame> _frames;

		//! Alignment for uniform buffers
		size_t _cbufferAlignment{ 64 };

		//! Current offset into the constant buffer
		size_t _cbufferOffset{ 0 };

		//! Current offset into the linear memory buffer
		size_t _linearBufferOffset{ 0 };

		//! \brief Buffer index used for push constants
		//! Uses the first available buffer as there is not upper limit
		const int _pushConstantBufferIndex{ 0 };

	private: // Command management
		//! Command lock
		std::mutex _cmdMutex;

		//! List of generic commands executed at frame start
		std::vector<std::function<void(void)>> _genericCmds;

	private: // Tracking state
		Frame* _currentFrame{ nullptr };

		// Currently active blend-state
		uint32_t _blendstate_hash{ 0 };

		//! The current primitive type to draw
		PrimitiveType _currentPrimitiveType;

		//! Number of vertices, if a patch is to be drawn
		int _currentNrPatchVertices;
	};
}}}}
