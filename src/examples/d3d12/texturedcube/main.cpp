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
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/d3d12/commandqueue.h>
#include <vcl/graphics/d3d12/descriptortable.h>
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
		: Application("SpinningCubes")
	{
		using Vcl::Graphics::D3D12::ContantDescriptor;
		using Vcl::Graphics::D3D12::DescriptorTableLayoutEntryType;
		using Vcl::Graphics::D3D12::DescriptorTableLayoutEntry;
		using Vcl::Graphics::D3D12::DescriptorTableLayout;
		using Vcl::Graphics::D3D12::DescriptorTable;
		using Vcl::Graphics::D3D12::TableDescriptor;
		using Vcl::Graphics::SurfaceFormat;
		using Vcl::Graphics::Runtime::D3D12::Buffer;
		using Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState;
		using Vcl::Graphics::Runtime::D3D12::RenderTargetLayout;
		using Vcl::Graphics::Runtime::D3D12::Shader;
		using Vcl::Graphics::Runtime::D3D12::Texture2D;
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferInitData;
		using Vcl::Graphics::Runtime::BufferUsage;
		using Vcl::Graphics::Runtime::CullModeMethod;
		using Vcl::Graphics::Runtime::InputLayoutDescription;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::PrimitiveType;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::Texture2DDescription;
		using Vcl::Graphics::Runtime::TextureResource;
		using Vcl::Graphics::Runtime::TextureUsage;
		using Vcl::Graphics::Runtime::VertexDataClassification;

		resetCommandList();

		_vs = std::make_unique<Shader>(ShaderType::VertexShader, 0, CubeCsoVS);
		_ps = std::make_unique<Shader>(ShaderType::FragmentShader, 0, CubeCsoPS);

		std::vector<DescriptorTableLayoutEntry> dynamic_resources =
		{
			{ DescriptorTableLayoutEntryType::Constant, ContantDescriptor{0, 0, 16}, D3D12_SHADER_VISIBILITY_VERTEX },
			{ DescriptorTableLayoutEntryType::Table, TableDescriptor{{
				{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND}
			}}, D3D12_SHADER_VISIBILITY_PIXEL }
		};
		std::vector<D3D12_STATIC_SAMPLER_DESC> static_samplers =
		{
			CD3DX12_STATIC_SAMPLER_DESC(
				0,
				D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
				D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
				0, 16,
				D3D12_COMPARISON_FUNC_LESS_EQUAL,
				D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
				0.f,
				D3D12_FLOAT32_MAX,
				D3D12_SHADER_VISIBILITY_PIXEL,
				0
			)
		};
		_tableLayout = std::make_unique<DescriptorTableLayout>(device(), std::move(dynamic_resources), std::move(static_samplers));

		InputLayoutDescription input_layout
		{
			{
				{ 0, 5*sizeof(float), VertexDataClassification::VertexDataPerObject },
			},
			{
				{ "Position", SurfaceFormat::R32G32B32_FLOAT, 0, 0,  0 },
				{ "UV",       SurfaceFormat::R32G32_FLOAT,    0, 0, 12 },
			}
		};

		PipelineStateDescription psd;
		psd.VertexShader = _vs.get();
		psd.FragmentShader = _ps.get();
		psd.InputAssembly.Topology = PrimitiveType::Trianglelist;
		psd.InputLayout = input_layout;
		RenderTargetLayout rtd = {};
		rtd.NumRenderTargets = 1;
		rtd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		rtd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
		_gps = std::make_unique<GraphicsPipelineState>(device(), psd, _tableLayout.get(), &rtd);

		_camera.setNearPlane(0.01f);
		_camera.setFarPlane(10.0f);
		_camera.setPosition({ 1.5f, 1.5f, 1.5f });

		std::vector<float> cube_points =
		{
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
		BufferDescription vbo_desc =
		{
			cube_points.size() * sizeof(float),
			BufferUsage::Vertex
		};
		BufferInitData vbo_data =
		{
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

		_table = std::make_unique<DescriptorTable>(device(), _tableLayout.get());
		_table->addResource(0, _tex.get());

		VCL_DIRECT3D_SAFE_CALL(cmdList()->Close());
		ID3D12CommandList* const generic_list = cmdList();
		device()->defaultQueue()->nativeQueue()->ExecuteCommandLists(1, &generic_list);
		device()->defaultQueue()->sync();
	}

private:
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
		using Vcl::Graphics::Runtime::D3D12::PipelineBindPoint;

		const auto size = swapChain()->bufferSize();
		const auto w = size.first;
		const auto h = size.second;
		auto cmd_list = cmd_buffer->handle();

		cmd_list->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
		D3D12_VIEWPORT viewport{ 0, 0, w, h, 0, 1 };
		cmd_list->RSSetViewports(1, &viewport);
		D3D12_RECT sr{ 0, 0, w, h };
		cmd_list->RSSetScissorRects(1, &sr);

		cmd_buffer->bindPipeline(_gps.get());
		cmd_buffer->bindDescriptorTable(PipelineBindPoint::Graphics, _table.get());
		cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		D3D12_VERTEX_BUFFER_VIEW vbv[] = {
			{ _vbo->handle()->GetGPUVirtualAddress(), _vbo->sizeInBytes(), 5*sizeof(float) }
		};
		cmd_list->IASetVertexBuffers(0, 1, vbv);

		Eigen::Affine3f rot{ Eigen::AngleAxisf{ _cubeRotation, Eigen::Vector3f::UnitY() } };
		Eigen::Matrix4f mvp = _camera.projection() * _camera.view() * rot.matrix();
		cmd_list->SetGraphicsRoot32BitConstants(0, 16, mvp.data(), 0);

		cmd_list->DrawInstanced(36, 1, 0, 0);
	}

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _vs;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _ps;

	std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTableLayout> _tableLayout;
	std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTable> _table;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState> _gps;

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Buffer> _vbo;

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Texture2D> _tex;

	Vcl::Graphics::Camera _camera{ std::make_shared<Vcl::Graphics::Direct3D::MatrixFactory>() };

	float _cubeRotation{ 0 };
};

int main(int argc, char** argv)
{
	SpinningCubeApplication app;
	return app.run();
}
