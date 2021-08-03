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

#include "../common/imguiapp.h"

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>
#include <vcl/graphics/camera.h>
#include <vcl/graphics/trackballcameracontroller.h>

//#include "shaders/boundinggrid.h"
#include "boundinggrid.vs.hlsl.cso.h"
#include "boundinggrid.gs.hlsl.cso.h"
#include "boundinggrid.ps.hlsl.cso.h"

bool InputUInt(const char* label, unsigned int* v, int step, int step_fast, ImGuiInputTextFlags flags)
{
	// Hexadecimal input provided as a convenience but the flag name is awkward. Typically you'd use InputText() to parse your own data, if you want to handle prefixes.
	const char* format = (flags & ImGuiInputTextFlags_CharsHexadecimal) ? "%08X" : "%d";
	return ImGui::InputScalar(label, ImGuiDataType_U32, (void*)v, (void*)(step > 0 ? &step : NULL), (void*)(step_fast > 0 ? &step_fast : NULL), format, flags);
}

struct TransformData
{
	// Transform to world space
	Eigen::Matrix4f ModelMatrix;

	// Transform from world to normalized device coordinates
	Eigen::Matrix4f ViewProjectionMatrix;
};

struct BoundingGridConfig
{
	// Axis' in model space
	Eigen::Vector4f Axis[3];

	// Colours of the box faces
	Eigen::Vector4f Colours[3];

	// Root position
	Eigen::Vector3f Origin;

	// Size of a single cell
	float StepSize;

	// Number of cells per size
	float Resolution;
};

class DynamicBoundingGridExample final : public ImGuiApplication
{
public:
	DynamicBoundingGridExample()
	: ImGuiApplication("Grid Cube")
	{
		using Vcl::Graphics::Camera;
		using Vcl::Graphics::SurfaceFormat;
		using Vcl::Graphics::D3D12::ContantDescriptor;
		using Vcl::Graphics::D3D12::DescriptorTable;
		using Vcl::Graphics::D3D12::DescriptorTableLayout;
		using Vcl::Graphics::D3D12::DescriptorTableLayoutEntry;
		using Vcl::Graphics::D3D12::DescriptorTableLayoutEntryType;
		using Vcl::Graphics::D3D12::InlineDescriptor;
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferUsage;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::PrimitiveType;
		using Vcl::Graphics::Runtime::RenderTargetLayout;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::D3D12::Buffer;
		using Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState;
		using Vcl::Graphics::Runtime::D3D12::Shader;

		_camera = std::make_unique<Camera>(std::make_shared<Vcl::Graphics::OpenGL::MatrixFactory>());

		_cameraController = std::make_unique<Vcl::Graphics::TrackballCameraController>();
		_cameraController->setCamera(_camera.get());

		std::vector<DescriptorTableLayoutEntry> dynamic_resources =
		{
			{ DescriptorTableLayoutEntryType::InlineConstantBufferView, InlineDescriptor{0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE}, D3D12_SHADER_VISIBILITY_VERTEX },
			{ DescriptorTableLayoutEntryType::InlineConstantBufferView, InlineDescriptor{1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE}, D3D12_SHADER_VISIBILITY_VERTEX }
		};
		_tableLayout = std::make_unique<DescriptorTableLayout>(device(), std::move(dynamic_resources));
		//_table = std::make_unique<DescriptorTable>(device(), _tableLayout.get());

		Shader boxVert{ ShaderType::VertexShader, 0, GridCsoVS };
		Shader boxGeom{ ShaderType::GeometryShader, 0, GridCsoGS };
		Shader boxFrag{ ShaderType::FragmentShader, 0, GridCsoPS };
		PipelineStateDescription boxPSDesc;
		boxPSDesc.VertexShader = &boxVert;
		boxPSDesc.GeometryShader = &boxGeom;
		boxPSDesc.FragmentShader = &boxFrag;
		boxPSDesc.InputAssembly.Topology = PrimitiveType::LinelistAdj;
		RenderTargetLayout rtd = {};
		rtd.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
		rtd.DepthStencilFormat = SurfaceFormat::D32_FLOAT;
		_boxPipelineState = std::make_unique<GraphicsPipelineState>(device(), boxPSDesc, rtd, _tableLayout.get());

		// Allocate a junk of 512 KB for constant buffers per frame
		BufferDescription cbuffer_desc;
		cbuffer_desc.SizeInBytes = 1 << 19;
		cbuffer_desc.Usage = BufferUsage::MapWrite | BufferUsage::Uniform;

		_constantBuffer[0] = Vcl::make_owner<Buffer>(device(), cbuffer_desc);
		_constantBuffer[1] = Vcl::make_owner<Buffer>(device(), cbuffer_desc);
		_constantBuffer[2] = Vcl::make_owner<Buffer>(device(), cbuffer_desc);
		_mappedConstantBuffer[0] = (uint8_t*)_constantBuffer[0]->map();
		_mappedConstantBuffer[1] = (uint8_t*)_constantBuffer[1]->map();
		_mappedConstantBuffer[2] = (uint8_t*)_constantBuffer[2]->map();
	}

private:
	void createDeviceObjects() override
	{
		Application::createDeviceObjects();

		const auto size = swapChain()->bufferSize();
		_camera->setViewport(size.first, size.second);
		_camera->setFieldOfView((float)size.first / (float)size.second);
		_camera->encloseInFrustum({ 0, 0, 0 }, { 0, -1, 0 }, 15.0f, { 0, 0, 1 });
	}

	void updateFrame() override
	{
		ImGuiApplication::updateFrame();

		// Update UI
		ImGui::Begin("Grid parameters");
		InputUInt("Resolution", &_gridResolution, 1, 10, ImGuiInputTextFlags_None);
		ImGui::End();

		// Update camera
		ImGuiIO& io = ImGui::GetIO();
		const auto size = swapChain()->bufferSize();
		const auto x = io.MousePos.x;
		const auto y = io.MousePos.y;
		const auto w = size.first;
		const auto h = size.second;
		if (io.MouseClicked[0] && !io.WantCaptureMouse)
		{
			_cameraController->startRotate((float)x / (float)w, (float)y / (float)h);
		} else if (io.MouseDown[0])
		{
			_cameraController->rotate((float)x / (float)w, (float)y / (float)h);
		} else if (io.MouseReleased[0])
		{
			_cameraController->endRotate();
		}
	}

	void renderFrame(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv) override
	{
		using namespace Vcl::Graphics::Runtime;

		RenderPassDescription rp_desc = {};
		rp_desc.RenderTargetAttachments.resize(1);
		rp_desc.RenderTargetAttachments[0].Attachment = reinterpret_cast<void*>(rtv.ptr);
		rp_desc.RenderTargetAttachments[0].ClearColor = { 0, 0, 0, 1 };
		rp_desc.RenderTargetAttachments[0].LoadOp = AttachmentLoadOp::Clear;
		rp_desc.DepthStencilTargetAttachment.Attachment = reinterpret_cast<void*>(dsv.ptr);
		rp_desc.DepthStencilTargetAttachment.ClearDepth = 1.0f;
		rp_desc.DepthStencilTargetAttachment.DepthLoadOp = AttachmentLoadOp::Clear;
		cmd_buffer->beginRenderPass(rp_desc);

		const auto size = swapChain()->bufferSize();
		const auto w = size.first;
		const auto h = size.second;

		D3D12_VIEWPORT viewport{ 0, 0, w, h, 0, 1 };
		D3D12_RECT sr{ 0, 0, w, h };
		cmd_buffer->handle()->RSSetViewports(1, &viewport);
		cmd_buffer->handle()->RSSetScissorRects(1, &sr);

		Eigen::Matrix4f vp = _camera->projection() * _camera->view();
		Eigen::Matrix4f m = _cameraController->currObjectTransformation();
		Eigen::AlignedBox3f bb{ Eigen::Vector3f{ -10.0f, -10.0f, -10.0f }, Eigen::Vector3f{ 10.0f, 10.0f, 10.0f } };
		renderBoundingBox(cmd_buffer->handle(), bb, _gridResolution, _boxPipelineState.get(), m, vp);

		cmd_buffer->endRenderPass();

		ImGuiApplication::renderFrame(cmd_buffer, rtv, dsv);
	}

	void renderBoundingBox(
		ID3D12GraphicsCommandList* cmd_list,
		const Eigen::AlignedBox3f& bb,
		unsigned int resolution,
		Vcl::Graphics::Runtime::PipelineState* ps,
		const Eigen::Matrix4f& M,
		const Eigen::Matrix4f& VP)
	{
		// Configure the layout
		cmd_list->SetPipelineState(_boxPipelineState->handle());
		cmd_list->SetGraphicsRootSignature(_tableLayout->rootSignature());

		// View on the scene
		const auto curr = swapChain()->currentBufferIndex();
		auto cbuf_transform = reinterpret_cast<TransformData*>(_mappedConstantBuffer[curr]);
		cbuf_transform->ModelMatrix = M;
		cbuf_transform->ViewProjectionMatrix = VP;
		cmd_list->SetGraphicsRootConstantBufferView(0, _constantBuffer[curr]->handle()->GetGPUVirtualAddress());

		// Compute the grid paramters
		float maxSize = bb.diagonal().maxCoeff();
		Eigen::Vector3f origin = bb.center() - 0.5f * maxSize * Eigen::Vector3f::Ones().eval();

		auto cbuf_config = reinterpret_cast<BoundingGridConfig*>(_mappedConstantBuffer[curr] + 1024);
		cbuf_config->Axis[0] = { 1, 0, 0, 0 };
		cbuf_config->Axis[1] = { 0, 1, 0, 0 };
		cbuf_config->Axis[2] = { 0, 0, 1, 0 };
		cbuf_config->Colours[0] = { 1, 0, 0, 0 };
		cbuf_config->Colours[1] = { 0, 1, 0, 0 };
		cbuf_config->Colours[2] = { 0, 0, 1, 0 };
		cbuf_config->Origin = origin;
		cbuf_config->StepSize = maxSize / (float)resolution;
		cbuf_config->Resolution = (float)resolution;

		cmd_list->SetGraphicsRootConstantBufferView(1, _constantBuffer[curr]->handle()->GetGPUVirtualAddress() + 1024);

		// Render the grid
		// 3 Line-loops with 4 points, N+1 replications of the loops (N tiles)
		cmd_list->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_LINELIST_ADJ);
		cmd_list->DrawInstanced(12, resolution + 1, 0, 0);
	}

	std::unique_ptr<Vcl::Graphics::TrackballCameraController> _cameraController;
	std::unique_ptr<Vcl::Graphics::Camera> _camera;

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _vs;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _gs;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _ps;

	std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTableLayout> _tableLayout;
	std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTable> _table;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState> _boxPipelineState;

	std::array<Vcl::owner_ptr<Vcl::Graphics::Runtime::D3D12::Buffer>, 3> _constantBuffer;
	std::array<uint8_t*, 3> _mappedConstantBuffer;

	unsigned int _gridResolution{ 10 };
};

int main(int argc, char** argv)
{
	DynamicBoundingGridExample app;
	return app.run();
}
