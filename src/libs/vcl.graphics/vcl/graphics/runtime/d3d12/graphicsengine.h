/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
#include <mutex>
#include <vector>

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/core/span.h>
#include <vcl/graphics/d3d12/descriptortable.h>
#include <vcl/graphics/d3d12/device.h>
#include <vcl/graphics/d3d12/swapchain.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/resource/texture.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>

namespace Vcl { namespace Graphics { namespace Runtime 
{
	class BufferRange
	{
	public:
		BufferRange(ref_ptr<Buffer> buf, size_t offset, size_t size)
			: _owner(buf), _offsetInBytes(offset), _sizeInBytes(size)
		{
		}

		const Buffer& owner() const { return *_owner; }

		size_t offset() const { return _offsetInBytes; }
		size_t size() const { return _sizeInBytes; }

	protected:
		//! Buffer to which the view belongs
		ref_ptr<Buffer> _owner;

		//! Offset in bytes
		size_t _offsetInBytes{ 0 };

		//! Size in bytes
		size_t _sizeInBytes{ 0 };
	};

	class BufferView : public BufferRange
	{
	public:
		BufferView(ref_ptr<Buffer> buf, size_t offset, size_t size, void* base_data_ptr = nullptr, std::function<void(Buffer&, size_t, size_t)> release = nullptr)
			: BufferRange(buf, offset, size), _release(std::move(release))
		{
			if (base_data_ptr)
			{
				char* ptr = reinterpret_cast<char*>(base_data_ptr);
				ptr += offset;
				_data = ptr;
			}
		}
		~BufferView()
		{
			if (_release)
				_release(*_owner, _offsetInBytes, _sizeInBytes);
		}

		BufferView(const BufferView&) = delete;
		BufferView(BufferView&&) = default;
		BufferView& operator =(const BufferView&) = delete;
		BufferView& operator =(BufferView&&) = default;

		void* data() const { return _data; }

	protected:
		//! Pointer to mapped data
		void* _data{ nullptr };

		//! Release the buffer view
		std::function<void(Buffer&, size_t, size_t)> _release;
	};

	template<typename T>
	class ConstantBufferView : public BufferView
	{
	public:
		ConstantBufferView(BufferView&& view) : BufferView(std::move(view)) {}
		T* operator->() { return reinterpret_cast<T*>(data()); }
	};

	class GraphicsEngine
	{
	public:
		GraphicsEngine() = default;
		GraphicsEngine(const GraphicsEngine&) = delete;
		GraphicsEngine(GraphicsEngine&&) = delete;
		virtual ~GraphicsEngine() = default;

	public:
		//! \name ResourceAllocation Resource allocation
		//! \{
		//virtual owner_ptr<Texture> createResource(const Texture2DDescription& desc);
		//virtual owner_ptr<Buffer> createResource(const BufferDescription& desc);
		//! \}

	public:
		//! Begin a new frame
		virtual void beginFrame() = 0;

		//! End the current frame
		virtual void endFrame() = 0;

		//! Query the current frame index
		//! \returns The index of the current frame
		uint64_t currentFrame() const { return _currentFrame; }

		//! Request a new constant buffer for per frame data
		virtual BufferView requestPerFrameConstantBuffer(size_t size) = 0;

		//! Request linear device memory for per frame data
		virtual BufferView requestPerFrameLinearMemory(size_t size) = 0;

		//! Enque a read-back command which will be executed at beginning of the frame where the data is ready
		virtual void enqueueReadback(const Texture& tex, std::function<void(stdext::span<uint8_t>)> callback) = 0;

		//! Enque a generic command which will be executed next frame
		virtual void enqueueCommand(std::function<void(void)>) = 0;

	protected:
		void incrFrameCounter() { _currentFrame++; }

	private:
		int64_t _currentFrame{ -1 };
	};
}}}

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 
{
	struct Frame
	{
	public:
		Frame(ref_ptr<Graphics::D3D12::Device> dev);

		ref_ptr<Buffer> constantBuffer() const { return _constantBuffer; }
		void* mappedConstantBuffer() const { return _mappedConstantBuffer; }

		ref_ptr<Buffer> linearMemoryBuffer() const { return _linearMemoryBuffer; }
		void* mappedLinearMemoryBuffer() const { return _mappedLinearMemory; }

		void mapBuffers();
		void unmapBuffers();

		Microsoft::WRL::ComPtr<ID3D12CommandAllocator> GraphicsCommandAllocator;

		//! Buffer used for shader constant data
		owner_ptr<Buffer> _constantBuffer;

		/// Pointer to mapped constant buffer
		void* _mappedConstantBuffer{ nullptr };

		//! Buffer used for linear memory
		owner_ptr<Buffer> _linearMemoryBuffer;

		/// Pointer to mapped constant buffer
		void* _mappedLinearMemory{ nullptr };

	};

	struct DepthSurface
	{
		void create(ID3D12Device* dev, uint32_t width, uint32_t height);
		void invalidate();

		Microsoft::WRL::ComPtr<ID3D12Resource> DepthBuffer;
		Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> DsvHeap;

	};

	enum class PipelineBindPoint
	{
		Graphics,
		Compute,
		RayTracing
	};

	class CommandBuffer final
	{
	public:
		using Device = Graphics::D3D12::Device;

		CommandBuffer(ref_ptr<Device> device, ID3D12CommandAllocator* allocator);

		ID3D12GraphicsCommandList* handle() { return _cmdList.Get(); }

		//! Reset the command-buffer to reuse with a new allocator
		void reset(Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator);

		void bindPipeline(PipelineState* pipeline);
		void bindDescriptorTable(PipelineBindPoint bp, uint32_t root_index, Graphics::D3D12::DescriptorTable* table);
		void bindDescriptorTables(PipelineBindPoint bp, uint32_t root_index, stdext::span<Graphics::D3D12::DescriptorTable*> tables);

		void bindIndexBuffer(Buffer* buffer);
		void bindVertexBuffer(Buffer* buffer);
		void bindVertexBuffers(stdext::span<Buffer*> buffers);

		void draw(
			unsigned int vertex_count_per_instance,
			unsigned int instance_count,
			unsigned int start_vertex_location,
			unsigned int start_instance_location);

		void drawIndexed(
			unsigned int index_count_per_instance,
			unsigned int instance_count,
			unsigned int start_index_location,
			int base_vertex_location,
			unsigned int start_instance_location);

		void dispatch(
			unsigned int thread_group_count_x,
			unsigned int thread_group_count_y,
			unsigned int thread_group_count_z);

	private:
		Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> _cmdList;

		PipelineState* _currentGraphicsPipeline{ nullptr };
	};

	class GraphicsEngine final : public Runtime::GraphicsEngine
	{
	public:
		using Device = Graphics::D3D12::Device;
		using SwapChainDescription = Graphics::D3D12::SwapChainDescription;
		using SwapChain = Graphics::D3D12::SwapChain;

		GraphicsEngine(ref_ptr<Device> device, const SwapChainDescription& swap_chain_desc);
		virtual ~GraphicsEngine();

		void beginFrame() override;
		void endFrame() override;

		BufferView requestPerFrameConstantBuffer(size_t size) override;
		BufferView requestPerFrameLinearMemory(size_t size) override;
		void enqueueReadback(const Runtime::Texture& tex, std::function<void(stdext::span<uint8_t>)> callback) override;
		void enqueueCommand(std::function<void(void)>) override;

	private:
		//! Associated D3D12 device
		ref_ptr<Device> _device;

		//! Swap chain used for presenting images
		std::unique_ptr<SwapChain> _swapChain;

		//! Command-list to submit engine internal commands
		Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> _graphicsCommandList;

		//! Depth-surface for depth-tests
		DepthSurface _depthSurface;

		//! \name Frames Frame data
		//! \{
		//! Number of parallel frames
		const int _numConcurrentFrames{ 3 };

		//! Per frame configuration data
		//! Create three frames:
		//! 1. where the CPU writes to
		//! 2. which the GPU driver is processing
		//! 3. which the GPU is processing
		std::vector<Frame> _frames;

		//! Current frame
		Frame* _currentFrame{ nullptr };
		//! \}

		//! \name Commands Arbitrary command execution
		//! \{
		//! Command lock
		std::mutex _cmdMutex;

		//! List of generic commands executed at frame start
		std::vector<std::function<void(void)>> _genericCmds;
		//! \}

		//! Alignment for uniform buffers
		size_t _cbufferAlignment{ 256 };

		//! Current offset into the constant buffer
		size_t _cbufferOffset{ 0 };

		//! Current offset into the linear memory buffer
		size_t _linearBufferOffset{ 0 };

	};
}}}}
