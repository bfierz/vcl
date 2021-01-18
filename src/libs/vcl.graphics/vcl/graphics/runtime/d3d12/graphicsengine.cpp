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
#include <vcl/graphics/runtime/d3d12/graphicsengine.h>

// Abseil
#include <absl/container/inlined_vector.h>

 // VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12 
{
	Frame::Frame(ref_ptr<Graphics::D3D12::Device> dev)
	{
		GraphicsCommandAllocator = dev->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);

		// Allocate a junk of 512 KB for constant buffers per frame
		BufferDescription cbuffer_desc;
		cbuffer_desc.SizeInBytes = 1 << 19;
		cbuffer_desc.Usage = BufferUsage::MapWrite | BufferUsage::Uniform;

		_constantBuffer = make_owner<Buffer>(dev.get(), cbuffer_desc);

		// Allocate a junk of 16 MB for linear memory per frame
		BufferDescription linbuffer_desc;
		linbuffer_desc.SizeInBytes = 1 << 24;
		linbuffer_desc.Usage = BufferUsage::MapWrite | BufferUsage::Vertex | BufferUsage::Index;

		_linearMemoryBuffer = make_owner<Buffer>(dev.get(), linbuffer_desc);
	}

	void Frame::mapBuffers()
	{
		_mappedConstantBuffer = _constantBuffer->map();
		_mappedLinearMemory = _linearMemoryBuffer->map();
	}

	void Frame::unmapBuffers()
	{
		_linearMemoryBuffer->unmap();
		_constantBuffer->unmap();
	}

	void DepthSurface::create(ID3D12Device* dev, uint32_t width, uint32_t height)
	{
		D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
		dsvHeapDesc.NumDescriptors = 1;
		dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
		dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		VCL_DIRECT3D_SAFE_CALL(dev->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&DsvHeap)));

		D3D12_CLEAR_VALUE optimizedClearValue = {};
		optimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
		optimizedClearValue.DepthStencil = { 1.0f, 0 };

		const CD3DX12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE_DEFAULT);
		const auto tex2d_desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height,
			1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
		VCL_DIRECT3D_SAFE_CALL(dev->CreateCommittedResource(
			&heap_props,
			D3D12_HEAP_FLAG_NONE,
			&tex2d_desc,
			D3D12_RESOURCE_STATE_DEPTH_WRITE,
			&optimizedClearValue,
			IID_PPV_ARGS(&DepthBuffer)
		));

		D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
		dsv.Format = DXGI_FORMAT_D32_FLOAT;
		dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
		dsv.Texture2D.MipSlice = 0;
		dsv.Flags = D3D12_DSV_FLAG_NONE;

		dev->CreateDepthStencilView(DepthBuffer.Get(), &dsv,
			DsvHeap->GetCPUDescriptorHandleForHeapStart());
	}
	void DepthSurface::invalidate()
	{
		DsvHeap.Reset();
		DepthBuffer.Reset();
	}

	D3D12_RENDER_PASS_ENDING_ACCESS_TYPE convert(AttachmentStoreOp op)
	{
		switch (op)
		{
		case AttachmentStoreOp::Clear:
			return D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_DISCARD;
		case AttachmentStoreOp::Store:
			return D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
		}
	}

	D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE convert(AttachmentLoadOp op)
	{
		switch (op)
		{
		case AttachmentLoadOp::DontCare:
			return D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_DISCARD;
		case AttachmentLoadOp::Clear:
			return D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
		case AttachmentLoadOp::Load:
			return D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE;
		}
	}

	CommandBuffer::CommandBuffer(ref_ptr<Device> device, ID3D12CommandAllocator* allocator)
	{
		_cmdList = device->createCommandList(allocator, D3D12_COMMAND_LIST_TYPE_DIRECT);
		VCL_DIRECT3D_SAFE_CALL(_cmdList->Close());
	}

	void CommandBuffer::reset(Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator)
	{
		_cmdList->Reset(allocator.Get(), nullptr);
	}

	void CommandBuffer::beginRenderPass(const RenderPassDescription& desc)
	{
		static_assert(sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) == sizeof(void*), "Texture view size is compatible");

		std::array<D3D12_RENDER_PASS_RENDER_TARGET_DESC, 8> rts;
		for (size_t i = 0; i < desc.RenderTargetAttachments.size(); i++)
		{
			rts[i].cpuDescriptor.ptr = reinterpret_cast<SIZE_T>(desc.RenderTargetAttachments[i].Attachment);
			rts[i].BeginningAccess.Type = convert(desc.RenderTargetAttachments[i].LoadOp);
			rts[i].BeginningAccess.Clear.ClearValue.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			std::copy(std::begin(desc.RenderTargetAttachments[i].ClearColor), std::end(desc.RenderTargetAttachments[i].ClearColor), rts[i].BeginningAccess.Clear.ClearValue.Color);
			rts[i].EndingAccess.Type = convert(desc.RenderTargetAttachments[i].StoreOp);
		}
		D3D12_RENDER_PASS_DEPTH_STENCIL_DESC dsv = {};
		dsv.cpuDescriptor.ptr = reinterpret_cast<SIZE_T>(desc.DepthStencilTargetAttachment.Attachment);
		dsv.DepthBeginningAccess.Type = convert(desc.DepthStencilTargetAttachment.DepthLoadOp);
		dsv.DepthBeginningAccess.Clear.ClearValue.Format = DXGI_FORMAT_D32_FLOAT;
		dsv.DepthBeginningAccess.Clear.ClearValue.DepthStencil.Depth = desc.DepthStencilTargetAttachment.ClearDepth;
		dsv.DepthEndingAccess.Type = convert(desc.DepthStencilTargetAttachment.DepthStoreOp);
		dsv.StencilBeginningAccess.Type = convert(desc.DepthStencilTargetAttachment.StencilLoadOp);
		dsv.StencilBeginningAccess.Clear.ClearValue.Format = DXGI_FORMAT_UNKNOWN;
		dsv.StencilBeginningAccess.Clear.ClearValue.DepthStencil.Stencil = desc.DepthStencilTargetAttachment.ClearStencil;
		dsv.StencilEndingAccess.Type = convert(desc.DepthStencilTargetAttachment.StencilStoreOp);

		auto dsv_ptr = desc.DepthStencilTargetAttachment.Attachment ? &dsv : nullptr;
		_cmdList->BeginRenderPass(desc.RenderTargetAttachments.size(), rts.data(), dsv_ptr, D3D12_RENDER_PASS_FLAG_NONE);
	}
	void CommandBuffer::endRenderPass()
	{
		_cmdList->EndRenderPass();
	}

	void CommandBuffer::bindPipeline(PipelineState* pipeline)
	{
		VclRequire(pipeline, "'pipeline' is set");
		VclRequire(dynamic_cast<GraphicsPipelineState*>(pipeline) || dynamic_cast<ComputePipelineState*>(pipeline),
			"'pipeline' is either graphics or compute pipeline");

		if (dynamic_cast<GraphicsPipelineState*>(pipeline))
		{
			_currentGraphicsPipeline = pipeline;
			_cmdList->SetPipelineState(static_cast<GraphicsPipelineState*>(pipeline)->handle());
		}
		else if (dynamic_cast<ComputePipelineState*>(pipeline))
				_cmdList->SetPipelineState(static_cast<ComputePipelineState*>(pipeline)->handle());
	}
	void CommandBuffer::bindDescriptorTable(PipelineBindPoint bp, uint32_t root_index, Graphics::D3D12::DescriptorTable* table)
	{
		std::array<Graphics::D3D12::DescriptorTable*, 1> tables = { table };
		bindDescriptorTables(bp, root_index, stdext::make_span(tables));
	}
	void CommandBuffer::bindDescriptorTables(PipelineBindPoint bp, uint32_t root_index, stdext::span<Graphics::D3D12::DescriptorTable*> tables)
	{
		VclRequire(std::all_of(tables.begin(), tables.end(), [](auto* table) { return table->layout(); }), "All tables bind to the same layout.");
		if (tables.empty())
			return;

		if (bp == PipelineBindPoint::Graphics)
		{
			tables[0]->setToGraphics(_cmdList.Get(), root_index);
		}
		else if (bp == PipelineBindPoint::Compute)
		{
			tables[0]->setToCompute(_cmdList.Get(), root_index);
		}
		else
		{
			VclDebugError("Pipeline bind point not implemented");
		}
	}
	void CommandBuffer::bindIndexBuffer(Buffer* buffer)
	{
		VclCheck(buffer->usage().isSet(BufferUsage::Index), "Resource requires index buffer binding setting");

		buffer->transition(_cmdList.Get(), D3D12_RESOURCE_STATE_INDEX_BUFFER);

		D3D12_INDEX_BUFFER_VIEW ibv = {};
		ibv.BufferLocation = buffer->handle()->GetGPUVirtualAddress();
		ibv.SizeInBytes = buffer->sizeInBytes();
		ibv.Format = DXGI_FORMAT_R32_UINT;
		_cmdList->IASetIndexBuffer(&ibv);
	}
	void CommandBuffer::bindVertexBuffer(Buffer* buffer)
	{
		std::array<Buffer*, 1> buffers = { buffer };
		bindVertexBuffers(stdext::make_span(buffers));
	}
	void CommandBuffer::bindVertexBuffers(stdext::span<Buffer*> buffers)
	{
		// Transition the resources that are bound
		for (auto* buffer : buffers)
		{
			VclCheck(buffer->resourcesStates() & D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "Resource requires vertex buffer binding setting");
			buffer->transition(_cmdList.Get(), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		}

		const auto* pps = static_cast<GraphicsPipelineState*>(_currentGraphicsPipeline);
		int idx = 0;
		absl::InlinedVector<D3D12_VERTEX_BUFFER_VIEW, 16> vbos(buffers.size());
		std::transform(buffers.begin(), buffers.end(), vbos.begin(), [pps, &idx](Buffer* buffer)
			{
				const auto stride = pps->inputLayout().binding(idx++).Stride;
				return D3D12_VERTEX_BUFFER_VIEW{ buffer->handle()->GetGPUVirtualAddress(), static_cast<unsigned int>(buffer->sizeInBytes()), stride };
			});

		_cmdList->IASetVertexBuffers(0, vbos.size(), vbos.data());
	}

	void CommandBuffer::draw(
		unsigned int vertex_count_per_instance,
		unsigned int instance_count,
		unsigned int start_vertex_location,
		unsigned int start_instance_location)
	{
		_cmdList->DrawInstanced(vertex_count_per_instance, instance_count, start_vertex_location, start_instance_location);
	}

	void CommandBuffer::drawIndexed(
		unsigned int index_count_per_instance,
		unsigned int instance_count,
		unsigned int start_index_location,
		int base_vertex_location,
		unsigned int start_instance_location)
	{
		_cmdList->DrawIndexedInstanced(index_count_per_instance, instance_count, start_index_location, base_vertex_location, start_instance_location);
	}

	void CommandBuffer::dispatch(
		unsigned int thread_group_count_x,
		unsigned int thread_group_count_y,
		unsigned int thread_group_count_z)
	{
		_cmdList->Dispatch(thread_group_count_x, thread_group_count_y, thread_group_count_z);
	}


	GraphicsEngine::GraphicsEngine
	(
		ref_ptr<Device> device,
		const SwapChainDescription& swap_chain_desc
	)
	: _device{device}
	{
		_swapChain = std::make_unique<SwapChain>(_device.get(), _device->defaultQueue(), swap_chain_desc);

		_frames.reserve(_numConcurrentFrames);
		for (int i = 0; i < _numConcurrentFrames; i++)
			_frames.emplace_back(_device);

		_graphicsCommandList = _device->createCommandList(_frames[0].GraphicsCommandAllocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
		VCL_DIRECT3D_SAFE_CALL(_graphicsCommandList->Close());

		if (swap_chain_desc.Width != 0 && swap_chain_desc.Height != 0)
		{
			_depthSurface.create(_device->nativeDevice(), swap_chain_desc.Width, swap_chain_desc.Height);
		}
	}
	GraphicsEngine::~GraphicsEngine()
	{
		_swapChain->wait();
	}

	void GraphicsEngine::beginFrame()
	{
		// Increment the frame counter to indicate the start of the next frame
		incrFrameCounter();

		// Wait until the requested frame is done with processing
		const auto frame_idx = _swapChain->waitForNextFrame();
		_currentFrame = &_frames[frame_idx];
		_currentFrame->GraphicsCommandAllocator->Reset();
		_graphicsCommandList->Reset(_frames[frame_idx].GraphicsCommandAllocator.Get(), nullptr);
		auto rtv = _swapChain->prepareFrame(_graphicsCommandList.Get());
		auto dsv = _depthSurface.DsvHeap->GetCPUDescriptorHandleForHeapStart();

		// Reset the staging area to use it again for the new frame
		//_currentFrame->executeReadbackRequests();

		// Map the per frame constant buffer
		_currentFrame->mapBuffers();

		// Reset the begin pointer for buffer chunk requests
		_cbufferOffset = 0;
		_linearBufferOffset = 0;

		// Execute the stored commands
		{
			std::unique_lock<std::mutex> lck{ _cmdMutex };

			for (auto& cmd : _genericCmds)
			{
				cmd();
			}

			_genericCmds.clear();
		}
	}
	void GraphicsEngine::endFrame()
	{
		// Unmap the constant buffer
		_currentFrame->unmapBuffers();

		// Clear the current frame to mark that not frame is active anymore
		_currentFrame = nullptr;

		// Flush the submitted commands and present to screen
		_swapChain->present(_device->defaultQueue(), _graphicsCommandList.Get(), false);
	}

	BufferView GraphicsEngine::requestPerFrameConstantBuffer(size_t size)
	{
		BufferView view{ _currentFrame->constantBuffer(), _cbufferOffset, size, _currentFrame->mappedConstantBuffer() };

		// Calculate the next offset
		size_t aligned_size = ((size + (_cbufferAlignment - 1)) / _cbufferAlignment) * _cbufferAlignment;
		_cbufferOffset += aligned_size;

		return view;
	}

	BufferView GraphicsEngine::requestPerFrameLinearMemory(size_t size)
	{
		BufferView view{ _currentFrame->linearMemoryBuffer(), _linearBufferOffset, size, _currentFrame->mappedLinearMemoryBuffer() };

		// Calculate the next offset
		size_t aligned_size = ((size + (_cbufferAlignment - 1)) / _cbufferAlignment) * _cbufferAlignment;
		_linearBufferOffset += aligned_size;

		return view;
	}

	void GraphicsEngine::enqueueReadback(const Runtime::Texture& tex, std::function<void(stdext::span<uint8_t>)> callback)
	{

	}
	void GraphicsEngine::enqueueCommand(std::function<void(void)>)
	{

	}
}}}}
