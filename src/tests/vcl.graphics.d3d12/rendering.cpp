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

 // VCL configuration
#include <vcl/config/global.h>

 // Google test
#include <gtest/gtest.h>

// Windows
#include <windows.h>

// Include the relevant parts from the library
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/d3d12/commandqueue.h>
#include <vcl/graphics/d3d12/descriptortable.h>
#include <vcl/graphics/d3d12/swapchain.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>

#include "quad.vs.hlsl.cso.h"
#include "quad.ps.hlsl.cso.h"

extern std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

class D3D12RenderingTest : public testing::Test
{
public:
	void SetUp() override
	{
		WNDCLASS wc = { 0 };
		wc.lpfnWndProc = WndProc;
		wc.hInstance = GetModuleHandle(NULL);
		wc.hbrBackground = (HBRUSH)(COLOR_BACKGROUND);
		wc.lpszClassName = "D3D12WindowClass";
		wc.style = CS_OWNDC;
		RegisterClass(&wc);
		_window_handle = CreateWindowEx(0, wc.lpszClassName, "D3D12Window", 0, 0, 0, 0, 0, HWND_MESSAGE, 0, 0, 0);
	}

	void TearDown() override
	{
		::CloseWindow(_window_handle);
		::UnregisterClass("D3D12WindowClass", GetModuleHandle(nullptr));
	}

protected:
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
	{
		switch (message)
		{
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		return 0;
	}

	void createDepthBufferAndView(ID3D12Device* dev, UINT width, UINT height)
	{
		D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
		dsvHeapDesc.NumDescriptors = 1;
		dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
		dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		VCL_DIRECT3D_SAFE_CALL(dev->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&_dsvHeap)));

		D3D12_CLEAR_VALUE optimizedClearValue = {};
		optimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
		optimizedClearValue.DepthStencil = { 1.0f, 0 };

		VCL_DIRECT3D_SAFE_CALL(dev->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height,
				1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
			D3D12_RESOURCE_STATE_DEPTH_WRITE,
			&optimizedClearValue,
			IID_PPV_ARGS(&_depthBuffer)
		));

		D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
		dsv.Format = DXGI_FORMAT_D32_FLOAT;
		dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
		dsv.Texture2D.MipSlice = 0;
		dsv.Flags = D3D12_DSV_FLAG_NONE;

		dev->CreateDepthStencilView(_depthBuffer.Get(), &dsv,
			_dsvHeap->GetCPUDescriptorHandleForHeapStart());
	}

	//! Native window handle of the test window
	HWND _window_handle;

	Microsoft::WRL::ComPtr<ID3D12Resource> _depthBuffer;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> _dsvHeap;
};

TEST_F(D3D12RenderingTest, RenderQuadWithoutData)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;

	// Swap chain as cheap mean to create a backbuffer
	SwapChainDescription desc;
	desc.Surface = _window_handle;
	desc.NumberOfImages = 2;
	desc.ColourFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.Width = 512;
	desc.Height = 512;
	desc.PresentMode = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	desc.VSync = false;

	SwapChain swap_chain{ device.get(), device->defaultQueue(), desc };
	createDepthBufferAndView(device->nativeDevice(), desc.Width, desc.Height);

	BufferDescription read_back_desc =
	{
		desc.Width* desc.Height * 4,
		BufferUsage::MapRead | BufferUsage::CopyDst
	};
	D3D12::Buffer read_back(device.get(), read_back_desc);
	EXPECT_TRUE(read_back.handle());

	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);

	D3D12::Shader vs(ShaderType::VertexShader, 0, QuadCsoVS);
	D3D12::Shader ps(ShaderType::FragmentShader, 0, QuadCsoPS);

	DescriptorTableLayout table_layout{ device.get(), {}, {} };
	auto signature = table_layout.rootSignature();

	PipelineStateDescription psd;
	psd.VertexShader = &vs;
	psd.FragmentShader = &ps;
	psd.InputAssembly.Topology = PrimitiveType::Trianglelist;
	psd.Rasterizer.CullMode = Vcl::Graphics::Runtime::CullModeMethod::None;
	D3D12::RenderTargetLayout rtd = {};
	rtd.NumRenderTargets = 1;
	rtd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	rtd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	D3D12::GraphicsPipelineState gps{device.get(), psd, &table_layout, &rtd};

	// Prepare a blank screen
	swap_chain.waitForNextFrame();
	const auto rtv = swap_chain.prepareFrame(cmd_list.Get());
	auto dsv = _dsvHeap->GetCPUDescriptorHandleForHeapStart();
	float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	cmd_list->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
	cmd_list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
	cmd_list->OMSetRenderTargets(1, &rtv, TRUE, &dsv);

	cmd_list->SetGraphicsRootSignature(signature);
	D3D12_VIEWPORT viewport{ 0, 0, 512, 512, 0, 1 };
	cmd_list->RSSetViewports(1, &viewport);
	D3D12_RECT sr{ 0, 0, 512, 512 };
	cmd_list->RSSetScissorRects(1, &sr);

	cmd_list->SetPipelineState(gps.handle());
	cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	cmd_list->DrawInstanced(6, 1, 0, 0);

	// Read back render target
	CD3DX12_RESOURCE_BARRIER barriers[] =
	{
		CD3DX12_RESOURCE_BARRIER::Transition(swap_chain.buffer(0), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)
	};
	cmd_list->ResourceBarrier(1, barriers);

	D3D12_PLACED_SUBRESOURCE_FOOTPRINT bufferFootprint = {};
	bufferFootprint.Footprint.Width = static_cast<UINT>(desc.Width);
	bufferFootprint.Footprint.Height = desc.Height;
	bufferFootprint.Footprint.Depth = 1;
	bufferFootprint.Footprint.RowPitch = static_cast<UINT>(desc.Width*4);
	bufferFootprint.Footprint.Format = desc.ColourFormat;

	CD3DX12_TEXTURE_COPY_LOCATION copyDest(read_back.handle(), bufferFootprint);
	CD3DX12_TEXTURE_COPY_LOCATION copySrc(swap_chain.buffer(0), 0);

	cmd_list->CopyTextureRegion(&copyDest, 0, 0, 0, &copySrc, nullptr);
	barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(swap_chain.buffer(0), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
	cmd_list->ResourceBarrier(1, barriers);


	VCL_DIRECT3D_SAFE_CALL(cmd_list->Close());
	ID3D12CommandList* const generic_lists[] = { cmd_list.Get() };
	device->defaultQueue()->nativeQueue()->ExecuteCommandLists(1, generic_lists);
	device->defaultQueue()->sync();

	cmd_list->Reset(cmd_allocator.Get(), nullptr);
	swap_chain.present(device->defaultQueue(), cmd_list.Get(), true);
	cmd_allocator->Reset();
	cmd_list->Reset(cmd_allocator.Get(), nullptr);

	bool equal = true;
	auto ptr = (std::array<uint8_t, 4>*)read_back.map({ 0, desc.Width * desc.Height * 4 });
	for (int i = 0; i < desc.Width * desc.Height; i++)
	{
		const bool r = ptr[i][0] == 255;
		const bool g = ptr[i][1] == 0;
		const bool b = ptr[i][2] == 255;
		const bool a = ptr[i][3] == 255;
		equal = equal && r && g && b && a;
	}
	read_back.unmap();

	// Verify the result
	EXPECT_TRUE(equal);
}
