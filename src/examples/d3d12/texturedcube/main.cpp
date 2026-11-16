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

#include "../common/app.h"

// Eigen
#include <Eigen/Geometry>

// VCL
#include <directx/d3dx12.h>
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/commandqueue.h>
#include <vcl/graphics/d3d12/swapchain.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>
#include <vcl/graphics/runtime/d3d12/resource/texture.h>
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>
#include <vcl/graphics/camera.h>
#include <vcl/math/math.h>

#include "../common/xortexture.h"

#include "cube.vs.hlsl.cso.h"
#include "cube.ps.hlsl.cso.h"

class SpinningCubeApplication final : public Application
{
public:
	SpinningCubeApplication()
	: Application("SpinningCubes", makeLayout())
	{
		using Vcl::Graphics::SurfaceFormat;
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferInitData;
		using Vcl::Graphics::Runtime::BufferUsage;
		using Vcl::Graphics::Runtime::CullModeMethod;
		using Vcl::Graphics::Runtime::InputLayoutDescription;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::PrimitiveType;
		using Vcl::Graphics::Runtime::RenderTargetLayout;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::Texture2DDescription;
		using Vcl::Graphics::Runtime::TextureResource;
		using Vcl::Graphics::Runtime::TextureUsage;
		using Vcl::Graphics::Runtime::VertexDataClassification;
		using Vcl::Graphics::Runtime::D3D12::Buffer;
		using Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState;
		using Vcl::Graphics::Runtime::D3D12::Shader;
		using Vcl::Graphics::Runtime::D3D12::Texture2D;

		resetCommandList();

		_vs = std::make_unique<Shader>(ShaderType::VertexShader, 0, CubeCsoVS);
		_ps = std::make_unique<Shader>(ShaderType::FragmentShader, 0, CubeCsoPS);

		// clang-format off
		InputLayoutDescription input_layout{
			{
				{ 0, 5 * sizeof(float), VertexDataClassification::VertexDataPerObject },
			},
			{
				{ "Position", SurfaceFormat::R32G32B32_FLOAT, 0, 0,  0 },
				{ "UV",       SurfaceFormat::R32G32_FLOAT,    0, 0, 12 },
			}
		};
		// clang-format on

		PipelineStateDescription psd;
		psd.VertexShader = _vs.get();
		psd.FragmentShader = _ps.get();
		psd.InputAssembly.Topology = PrimitiveType::Trianglelist;
		psd.InputLayout = input_layout;
		RenderTargetLayout rtd = {};
		rtd.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
		rtd.DepthStencilFormat = SurfaceFormat::D32_FLOAT;
		psd.RenderTargets = rtd;
		_gps = std::make_unique<GraphicsPipelineState>(device(), heap(), psd);

		_camera.setNearPlane(0.01f);
		_camera.setFarPlane(10.0f);
		_camera.setPosition({ 1.5f, 1.5f, 1.5f });

		// clang-format off
		std::vector<float> cube_points = {
			-1.0f,-1.0f,-1.0f, 0.0f, 0.0f,
			-1.0f,-1.0f, 1.0f, 0.0f, 1.0f,
			-1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-1.0f,-1.0f,-1.0f, 0.0f, 0.0f,
			-1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f,-1.0f, 1.0f, 0.0f,

			+1.0f,-1.0f, 1.0f, 1.0f, 1.0f,
			-1.0f,-1.0f,-1.0f, 0.0f, 0.0f,
			+1.0f,-1.0f,-1.0f, 1.0f, 0.0f,
			+1.0f,-1.0f, 1.0f, 1.0f, 1.0f,
			-1.0f,-1.0f, 1.0f, 0.0f, 1.0f,
			-1.0f,-1.0f,-1.0f, 0.0f, 0.0f,

			+1.0f, 1.0f,-1.0f, 1.0f, 1.0f,
			-1.0f,-1.0f,-1.0f, 0.0f, 0.0f,
			-1.0f, 1.0f,-1.0f, 0.0f, 1.0f,
			+1.0f, 1.0f,-1.0f, 1.0f, 1.0f,
			+1.0f,-1.0f,-1.0f, 1.0f, 0.0f,
			-1.0f,-1.0f,-1.0f, 0.0f, 0.0f,

			+1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			+1.0f,-1.0f,-1.0f, 0.0f, 0.0f,
			+1.0f, 1.0f,-1.0f, 1.0f, 0.0f,
			+1.0f,-1.0f,-1.0f, 0.0f, 0.0f,
			+1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			+1.0f,-1.0f, 1.0f, 0.0f, 1.0f,

			+1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			+1.0f, 1.0f,-1.0f, 1.0f, 0.0f,
			-1.0f, 1.0f,-1.0f, 0.0f, 0.0f,
			+1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f,-1.0f, 0.0f, 0.0f,
			-1.0f, 1.0f, 1.0f, 0.0f, 1.0f,

			-1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
			-1.0f,-1.0f, 1.0f, 0.0f, 0.0f,
			+1.0f,-1.0f, 1.0f, 1.0f, 0.0f,
			+1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
			+1.0f,-1.0f, 1.0f, 1.0f, 0.0f,
		};
		// clang-format on

		BufferDescription vbo_desc = {
			cube_points.size() * sizeof(float),
			BufferUsage::Vertex
		};
		BufferInitData vbo_data = {
			cube_points.data(),
			cube_points.size() * sizeof(float)
		};
		_vbo = std::make_unique<Buffer>(device(), vbo_desc, &vbo_data, cmdList());

		Texture2DDescription desc2d;
		desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
		desc2d.Usage = TextureUsage::Sampled;
		desc2d.ArraySize = 1;
		desc2d.Width = 512;
		desc2d.Height = 512;
		desc2d.MipLevels = 1;

		const auto image = createXorTexture(desc2d.Width, desc2d.Height);
		TextureResource res;
		res.Data = stdext::make_span(reinterpret_cast<const uint8_t*>(image.data()), image.size() * sizeof(decltype(image)::value_type));
		res.Width = desc2d.Width;
		res.Height = desc2d.Height;
		res.Format = desc2d.Format;

		_tex = std::make_unique<Texture2D>(device(), desc2d, &res, cmdList());

		_tex->transition(cmdList(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		_texView = heap()->createView(*_tex, Vcl::Graphics::Runtime::ResourceViewType::Texture);

		VCL_DIRECT3D_SAFE_CALL(cmdList()->Close());
		ID3D12CommandList* const generic_list = cmdList();
		device()->defaultQueue()->nativeQueue()->ExecuteCommandLists(1, &generic_list);
		device()->defaultQueue()->sync();
	}

private:
	//! Bindless layout with a linear-clamp sampler at register(s0, space1)
	static Vcl::Graphics::Runtime::BindlessLayoutDescription makeLayout()
	{
		using namespace Vcl::Graphics::Runtime;

		SamplerDescription linear_clamp;
		linear_clamp.Filter = FilterType::MinMagLinearMipPoint;
		linear_clamp.AddressU = TextureAddressMode::Clamp;
		linear_clamp.AddressV = TextureAddressMode::Clamp;
		linear_clamp.AddressW = TextureAddressMode::Clamp;
		linear_clamp.MinLOD = 0.0f;

		BindlessLayoutDescription layout;
		layout.StaticSamplers = { { 0, linear_clamp } };
		return layout;
	}

	void createDeviceObjects() override
	{
		Application::createDeviceObjects();

		const auto size = swapChain()->bufferSize();
		_camera.setViewport(size.first, size.second);
		_camera.setFieldOfView((float)size.first / (float)size.second);
	}

	void updateFrame() override
	{
		_cubeRotation += 0.01f;
		const float two_pi = 2 * Vcl::Mathematics::pi<float>();
		if (_cubeRotation > two_pi)
			_cubeRotation -= two_pi;
	}

	void renderFrame(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv) override
	{
		using namespace Vcl::Graphics::Runtime;
		using Vcl::Graphics::Runtime::D3D12::PipelineBindPoint;

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
		auto cmd_list = cmd_buffer->handle();

		D3D12_VIEWPORT viewport{ 0, 0, w, h, 0, 1 };
		cmd_list->RSSetViewports(1, &viewport);
		D3D12_RECT sr{ 0, 0, w, h };
		cmd_list->RSSetScissorRects(1, &sr);

		cmd_buffer->bindPipeline(_gps.get());
		cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		D3D12_VERTEX_BUFFER_VIEW vbv[] = {
			{ _vbo->handle()->GetGPUVirtualAddress(), _vbo->sizeInBytes(), 5 * sizeof(float) }
		};
		cmd_list->IASetVertexBuffers(0, 1, vbv);

		Eigen::Affine3f rot{ Eigen::AngleAxisf{ _cubeRotation, Eigen::Vector3f::UnitY() } };
		Eigen::Matrix4f mvp = _camera.projection() * _camera.view() * rot.matrix();
		cmd_buffer->pushConstants(PipelineBindPoint::Graphics, 0, sizeof(mvp), mvp.data());
		const uint32_t albedo_view = _texView.index().Value;
		cmd_buffer->pushConstants(PipelineBindPoint::Graphics, sizeof(mvp), sizeof(albedo_view), &albedo_view);

		cmd_list->DrawInstanced(36, 1, 0, 0);

		cmd_buffer->endRenderPass();
	}

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _vs;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _ps;

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState> _gps;

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Buffer> _vbo;

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Texture2D> _tex;
	Vcl::Graphics::D3D12::ResourceView _texView;

	Vcl::Graphics::Camera _camera{ std::make_shared<Vcl::Graphics::Direct3D::MatrixFactory>() };

	float _cubeRotation{ 0 };
};

int main(int argc, char** argv)
{
	SpinningCubeApplication app;
	return app.run();
}
