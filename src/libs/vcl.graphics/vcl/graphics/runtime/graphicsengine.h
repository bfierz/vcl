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

// C++ standard library
#include <memory>

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/resource/buffer.h>
#include <vcl/graphics/runtime/resource/texture.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/runtime/state/sampler.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	class BufferView
	{
	public:
		BufferView(ref_ptr<Buffer> buf, size_t offset, size_t size, void* base_data_ptr = nullptr)
		: _owner(buf), _offsetInBytes(offset), _sizeInBytes(size)
		{
			if (base_data_ptr)
			{
				char* ptr = reinterpret_cast<char*>(base_data_ptr);
				ptr += offset;
				_data = ptr;
			}
		}

	public:
		size_t offset() const { return _offsetInBytes; }
		size_t size() const { return _sizeInBytes; }

	public:
		void* data() const { return _data; }

	public:
		const Buffer& owner() const { return *_owner; }

	protected:
		//! Buffer to which the view belongs
		ref_ptr<Buffer> _owner;

		//! Offset in bytes
		size_t _offsetInBytes{ 0 };

		//! Size in bytes
		size_t _sizeInBytes{ 0 };

		//! Pointer to mapped data
		void* _data{ nullptr };
	};

	template<typename T>
	class ConstantBufferView
	{
	public:
		ConstantBufferView(BufferView&& view) : _view(std::move(view)) {}
		operator const BufferView&() const { return _view; }
		T* operator->() { return reinterpret_cast<T*>(_view.data()); }

	private:
		BufferView _view;
	};

	class GraphicsEngine
	{
	public:
		virtual ~GraphicsEngine() = default;

	public:
		//! \defgroup ResourceAllocation Resource allocation
		//! \{
		virtual owner_ptr<Texture> createResource(const Texture2DDescription& desc) =0;
		virtual owner_ptr<Buffer> createResource(const BufferDescription& desc) =0;
		//! \}

	public:
		//! Begin a new frame
		virtual void beginFrame() = 0;

		//! End the current frame
		virtual void endFrame() = 0;
		
		//! Query the current frame index
		//! \returns The index of the current frame
		uint64_t currentFrame() const { return _currentFrame; }

	public: // Dynamic resource allocation

		//! Request a new constant buffer for per frame data
		template<typename T>
		ConstantBufferView<T> requestPerFrameConstantBuffer()
		{
			return ConstantBufferView<T>(requestPerFrameConstantBuffer(sizeof(T)));
		}

		//! Request a new constant buffer for per frame data
		virtual BufferView requestPerFrameConstantBuffer(size_t size) = 0;

		//! Request linear device memory for per frame data
		virtual BufferView requestPerFrameLinearMemory(size_t size) = 0;

		//! Enque a read-back command which will be executed at beginning of the frame where the data is ready
		virtual void enqueueReadback(const Texture& tex, std::function<void(const BufferView&)> callback) = 0;

		//! Enque a generic command which will be executed next frame
		virtual void enqueueCommand(std::function<void(void)>) = 0;

	public: // Resource management
		void setRenderTargets(std::initializer_list<ref_ptr<Texture>> colour_targets, ref_ptr<Texture> depth_target)
		{
			setRenderTargets(std::span<const ref_ptr<Texture>>(colour_targets.begin(), colour_targets.size()), depth_target);
		}

		virtual void setRenderTargets(std::span<const ref_ptr<Texture>> colour_targets, ref_ptr<Texture> depth_target) = 0;

		virtual void setConstantBuffer(int idx, BufferView buffer) = 0;
		virtual void setVertexBuffer(int idx, const Buffer& buffer, int offset, int stride) = 0;
		virtual void setSampler(int idx, const Runtime::Sampler& sampler) = 0;
		virtual void setSamplers(int idx, std::span<const ref_ptr<Sampler>> samplers) = 0;
		virtual void setTexture(int idx, const Runtime::Texture& texture) = 0;
		virtual void setTextures(int idx, std::span<const ref_ptr<Texture>> textures) = 0;
		
		virtual void pushConstants(void* data, size_t size) = 0;

	public:
		//! \defgroup FramebufferCommands Framebuffer commands
		//! \{
		virtual void clear(int idx, const Eigen::Vector4f& colour) = 0;
		virtual void clear(int idx, const Eigen::Vector4i& colour) = 0;
		virtual void clear(int idx, const Eigen::Vector4ui& colour) = 0;
		virtual void clear(float depth, int stencil) = 0;
		virtual void clear(float depth) = 0;
		virtual void clear(int stencil) = 0;
		//! \}

	public: // Command buffer operations
		//! Set a new pipeline state
		virtual void setPipelineState(ref_ptr<PipelineState> state) = 0;
		
		//! \defgroup DrawCommannds Draw commands
		//! \{
		virtual void setPrimitiveType(PrimitiveType type, int nr_vertices = -1) = 0;
		virtual void draw(int count, int first = 0, int instance_count = 1, int base_instance = 0) = 0;
		virtual void drawIndexed(int count, int first_index = 0, int instance_count = 1, int base_vertex = 0, int base_instance = 0) = 0;
		//! \}

	protected:
		void incrFrameCounter() { _currentFrame++; }

	private:
		uint64_t _currentFrame{ 0 };
	};
}}}
