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

#include "imguiapp.h"

// VCL
#include <vcl/graphics/d3d12/d3d.h>

// IMGUI
#define NOMINMAX
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

ImGuiApplication::ImGuiApplication(LPCSTR title)
: Application(title)
{
	_imguiDescrHeap = device()->createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumberOfImGuiDescriptors, true);
	_imguiDescrSize = device()->nativeDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	for (UINT i = NumberOfImGuiDescriptors; i > 0; i--)
	{
		_imguiFreeDescrs.push_back(i - 1);
	}

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;	 // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;	  // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplWin32_Init(windowHandle());

	// The font atlas and every texture the application registers occupy one SRV descriptor each
	ImGui_ImplDX12_InitInfo init_info;
	init_info.Device = device()->nativeDevice();
	init_info.CommandQueue = device()->defaultQueue()->nativeQueue();
	init_info.NumFramesInFlight = NumberOfFrames;
	init_info.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	init_info.UserData = this;
	init_info.SrvDescriptorHeap = _imguiDescrHeap.Get();
	init_info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu) {
		static_cast<ImGuiApplication*>(info->UserData)->allocateDescriptor(*out_cpu, *out_gpu);
	};
	init_info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE gpu) {
		static_cast<ImGuiApplication*>(info->UserData)->freeDescriptor(cpu, gpu);
	};
	ImGui_ImplDX12_Init(&init_info);

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
	// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Read 'docs/FONTS.txt' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != NULL);
}

ImGuiApplication::~ImGuiApplication()
{
	ImGui_ImplDX12_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

void ImGuiApplication::allocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE& cpu, D3D12_GPU_DESCRIPTOR_HANDLE& gpu)
{
	IM_ASSERT(!_imguiFreeDescrs.empty());
	const UINT index = _imguiFreeDescrs.back();
	_imguiFreeDescrs.pop_back();

	cpu = Vcl::Graphics::D3D12::cpuHeapStart(_imguiDescrHeap.Get());
	cpu.ptr += static_cast<SIZE_T>(index) * _imguiDescrSize;
	gpu = Vcl::Graphics::D3D12::gpuHeapStart(_imguiDescrHeap.Get());
	gpu.ptr += static_cast<UINT64>(index) * _imguiDescrSize;
}

void ImGuiApplication::freeDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE)
{
	const D3D12_CPU_DESCRIPTOR_HANDLE start = Vcl::Graphics::D3D12::cpuHeapStart(_imguiDescrHeap.Get());
	_imguiFreeDescrs.push_back(static_cast<UINT>((cpu.ptr - start.ptr) / _imguiDescrSize));
}

void ImGuiApplication::updateFrame()
{
	ImGui_ImplDX12_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
}

LRESULT ImGuiApplication::msgHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
		return true;

	if (Application::msgHandler(hWnd, msg, wParam, lParam))
		return true;

	return false;
}
void ImGuiApplication::invalidateDeviceObjects()
{
	Application::invalidateDeviceObjects();

	ImGui_ImplDX12_InvalidateDeviceObjects();
}
void ImGuiApplication::createDeviceObjects()
{
	Application::createDeviceObjects();

	ImGui_ImplDX12_CreateDeviceObjects();
}

void ImGuiApplication::renderFrame(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv)
{
	using namespace Vcl::Graphics::Runtime;

	RenderPassDescription rp_desc = {};
	rp_desc.RenderTargetAttachments.resize(1);
	rp_desc.RenderTargetAttachments[0].Attachment = reinterpret_cast<void*>(rtv.ptr);
	rp_desc.RenderTargetAttachments[0].LoadOp = AttachmentLoadOp::Load;
	cmd_buffer->beginRenderPass(rp_desc);
	ID3D12DescriptorHeap* heaps[] = { imGuiDescriptorHeap() };
	cmd_buffer->handle()->SetDescriptorHeaps(1, heaps);
	ImGui::Render();
	ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_buffer->handle());
	cmd_buffer->endRenderPass();
}