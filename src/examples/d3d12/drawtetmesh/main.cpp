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
#include <vcl/geometry/meshfactory.h>
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/d3d12/commandqueue.h>
#include <vcl/graphics/d3d12/descriptortable.h>
#include <vcl/graphics/d3d12/swapchain.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>
#include <vcl/graphics/camera.h>
#include <vcl/math/math.h>

#include "tettotrimesh.cs.hlsl.cso.h"
#include "trimesh.vs.hlsl.cso.h"
#include "tetmesh.vs.hlsl.cso.h"
#include "tetmesh.gs.hlsl.cso.h"
#include "simple.ps.hlsl.cso.h"

class DrawTetMeshApplication final : public Application
{
	struct TetToTriMeshContext
	{
		std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTableLayout> TableLayoutCompute;
		std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTable> TableCompute;
		std::unique_ptr<Vcl::Graphics::Runtime::D3D12::ComputePipelineState> TetToTriMeshPS;

		std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTableLayout> TableLayoutGraphics;
		std::unique_ptr<Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState> RenderTriMeshPS;
	};

	struct DirectTetMeshContext
	{
		std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTableLayout> TableLayoutGraphics;
		std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTable> TableGraphics;
		std::unique_ptr<Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState> RenderTetMeshPS;
	};

public:
	DrawTetMeshApplication()
	: Application("TetMesh Rendering Application")
	{
		using namespace Vcl::Graphics::D3D12;
		using Vcl::Graphics::Runtime::D3D12::Buffer;
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferInitData;
		using Vcl::Graphics::Runtime::BufferUsage;

		resetCommandList();

		createMesh();
		createTetToTriMeshPipeline();
		createDirectTetMeshPipeline();

		_camera.setNearPlane(0.01f);
		_camera.setFarPlane(10.0f);
		_camera.setPosition({ 1.5f, 1.5f, 1.5f });

		VCL_DIRECT3D_SAFE_CALL(cmdList()->Close());
		ID3D12CommandList* const generic_list = cmdList();
		device()->defaultQueue()->nativeQueue()->ExecuteCommandLists(1, &generic_list);
		device()->defaultQueue()->sync();
	}

private:
	void createMesh()
	{
		using namespace Vcl::Geometry;
		using Vcl::Graphics::Runtime::D3D12::Buffer;
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferInitData;
		using Vcl::Graphics::Runtime::BufferUsage;

		_mesh = MeshFactory<TetraMesh>::createHomogenousCubes(1, 1, 1);

		BufferDescription vbo_desc =
		{
			_mesh->nrVertices() * sizeof(TetraMesh::Vertex),
			BufferUsage::Vertex | BufferUsage::Storage
		};
		BufferInitData vbo_data =
		{
			_mesh->vertices()->data(),
			_mesh->vertices()->size() * sizeof(TetraMesh::Vertex)
		};
		_tetMeshVertices = std::make_unique<Buffer>(device(), vbo_desc, &vbo_data, cmdList());

		BufferDescription ibo_desc =
		{
			_mesh->nrVolumes() * sizeof(TetraMesh::Volume),
			BufferUsage::Vertex | BufferUsage::Index | BufferUsage::Storage
		};
		BufferInitData ibo_data =
		{
			_mesh->volumes()->data(),
			_mesh->volumes()->size() * sizeof(TetraMesh::Volume)
		};
		_tetMeshIndices = std::make_unique<Buffer>(device(), ibo_desc, &ibo_data, cmdList());


		BufferDescription tri_vbo_desc =
		{
			4 * _mesh->nrVolumes() * sizeof(TetraMesh::Vertex),
			BufferUsage::Vertex | BufferUsage::Storage
		};
		_triMeshVertices = std::make_unique<Buffer>(device(), tri_vbo_desc);

		BufferDescription tri_ibo_desc =
		{
			3 * 4 * _mesh->nrVolumes() * sizeof(int),
			BufferUsage::Index | BufferUsage::Storage
		};
		_triMeshIndices = std::make_unique<Buffer>(device(), tri_ibo_desc);
	}

	void createTetToTriMeshPipeline()
	{
		using namespace Vcl::Geometry;
		using namespace Vcl::Graphics::D3D12;
		using Vcl::Graphics::Runtime::D3D12::ComputePipelineState;
		using Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState;
		using Vcl::Graphics::Runtime::D3D12::Shader;
		using Vcl::Graphics::Runtime::ComputePipelineStateDescription;
		using Vcl::Graphics::Runtime::InputLayoutDescription;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::PrimitiveType;
		using Vcl::Graphics::Runtime::RenderTargetLayout;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::VertexDataClassification;
		using Vcl::Graphics::SurfaceFormat;

		std::vector<DescriptorTableLayoutEntry> comp_dynamic_resources =
		{
			{ DescriptorTableLayoutEntryType::Constant, ContantDescriptor{0, 0, 2}, D3D12_SHADER_VISIBILITY_ALL },
			{ DescriptorTableLayoutEntryType::Table, TableDescriptor{{
				{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
				{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
				{ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
				{ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND}
			}}, D3D12_SHADER_VISIBILITY_ALL }
		};
		_tetToTriMeshCtx.TableLayoutCompute = std::make_unique<DescriptorTableLayout>(device(), std::move(comp_dynamic_resources));
		_tetToTriMeshCtx.TableCompute = std::make_unique<DescriptorTable>(device(), _tetToTriMeshCtx.TableLayoutCompute.get());
		_tetToTriMeshCtx.TableCompute->addResource(0, _tetMeshIndices.get(),  0, 4*_mesh->nrVolumes(), sizeof(int));
		_tetToTriMeshCtx.TableCompute->addResource(1, _tetMeshVertices.get(), 0, _mesh->nrVertices(), sizeof(TetraMesh::Vertex));
		_tetToTriMeshCtx.TableCompute->addResource(2, _triMeshIndices.get(),  0, 3 * 4 * _mesh->nrVolumes(), sizeof(int));
		_tetToTriMeshCtx.TableCompute->addResource(3, _triMeshVertices.get(), 0, 4 * _mesh->nrVolumes(), sizeof(TetraMesh::Vertex));

		Shader tet_to_tri_mesh_cs(ShaderType::ComputeShader, 0, TetToTriMeshCsoCS);
		ComputePipelineStateDescription cpsd;
		cpsd.ComputeShader = &tet_to_tri_mesh_cs;
		_tetToTriMeshCtx.TetToTriMeshPS = std::make_unique<ComputePipelineState>(device(), cpsd, _tetToTriMeshCtx.TableLayoutCompute.get());

		std::vector<DescriptorTableLayoutEntry> dynamic_resources =
		{
			{ DescriptorTableLayoutEntryType::Constant, ContantDescriptor{0, 0, 16}, D3D12_SHADER_VISIBILITY_VERTEX }
		};
		_tetToTriMeshCtx.TableLayoutGraphics = std::make_unique<DescriptorTableLayout>(device(), std::move(dynamic_resources));

		Shader tri_mesh_vs(ShaderType::VertexShader, 0, TriMeshCsoVS);
		Shader simple_ps(ShaderType::FragmentShader, 0, SimpleCsoPS);
		InputLayoutDescription input_layout
		{
			{
				{ 0, sizeof(Eigen::Vector3f), VertexDataClassification::VertexDataPerObject },
			},
			{
				{ "Position", SurfaceFormat::R32G32B32_FLOAT, 0, 0, 0 },
			}
		};

		PipelineStateDescription psd;
		psd.VertexShader = &tri_mesh_vs;
		psd.FragmentShader = &simple_ps;
		psd.InputAssembly.Topology = PrimitiveType::Trianglelist;
		psd.InputLayout = input_layout;
		RenderTargetLayout rtd = {};
		rtd.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
		rtd.DepthStencilFormat = SurfaceFormat::D32_FLOAT;
		_tetToTriMeshCtx.RenderTriMeshPS = std::make_unique<GraphicsPipelineState>(device(), psd, rtd, _tetToTriMeshCtx.TableLayoutGraphics.get());
	}

	void createDirectTetMeshPipeline()
	{
		using namespace Vcl::Geometry;
		using namespace Vcl::Graphics::D3D12;
		using Vcl::Graphics::Runtime::D3D12::ComputePipelineState;
		using Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState;
		using Vcl::Graphics::Runtime::D3D12::Shader;
		using Vcl::Graphics::Runtime::ComputePipelineStateDescription;
		using Vcl::Graphics::Runtime::InputLayoutDescription;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::PrimitiveType;
		using Vcl::Graphics::Runtime::RenderTargetLayout;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::VertexDataClassification;
		using Vcl::Graphics::SurfaceFormat;

		std::vector<DescriptorTableLayoutEntry> comp_dynamic_resources =
		{
			{ DescriptorTableLayoutEntryType::Constant, ContantDescriptor{0, 0, 16}, D3D12_SHADER_VISIBILITY_ALL },
			{ DescriptorTableLayoutEntryType::Table, TableDescriptor{{
				{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND}
			}}, D3D12_SHADER_VISIBILITY_ALL }
		};
		_directTetMeshCtx.TableLayoutGraphics = std::make_unique<DescriptorTableLayout>(device(), std::move(comp_dynamic_resources));
		_directTetMeshCtx.TableGraphics = std::make_unique<DescriptorTable>(device(), _directTetMeshCtx.TableLayoutGraphics.get());
		_directTetMeshCtx.TableGraphics->addResource(0, _tetMeshVertices.get(), 0, _mesh->nrVertices(), sizeof(TetraMesh::Vertex));

		Shader tet_mesh_vs(ShaderType::VertexShader, 0, TetMeshCsoVS);
		Shader tet_mesh_gs(ShaderType::GeometryShader, 0, TetMeshCsoGS);
		Shader simple_ps(ShaderType::FragmentShader, 0, SimpleCsoPS);
		InputLayoutDescription input_layout
		{
			{
				{ 0, sizeof(Eigen::Vector4i), VertexDataClassification::VertexDataPerObject },
			},
			{
				{ "Index", SurfaceFormat::R32G32B32A32_SINT, 0, 0, 0 },
			}
		};

		PipelineStateDescription psd;
		psd.VertexShader = &tet_mesh_vs;
		psd.GeometryShader = &tet_mesh_gs;
		psd.FragmentShader = &simple_ps;
		psd.InputAssembly.Topology = PrimitiveType::Pointlist;
		psd.InputLayout = input_layout;
		RenderTargetLayout rtd = {};
		rtd.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
		rtd.DepthStencilFormat = SurfaceFormat::D32_FLOAT;
		_directTetMeshCtx.RenderTetMeshPS = std::make_unique<GraphicsPipelineState>(device(), psd, rtd, _directTetMeshCtx.TableLayoutGraphics.get());
	}

	void renderFrameTetToTriMesh(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer)
	{
		using Vcl::Graphics::Runtime::D3D12::PipelineBindPoint;

		cmd_buffer->bindPipeline(_tetToTriMeshCtx.TetToTriMeshPS.get());
		cmd_buffer->bindDescriptorTable(PipelineBindPoint::Compute, _tetToTriMeshCtx.TableCompute.get());

		struct TetToTriMeshConversionParameters
		{
			// Number of tet indices
			uint32_t NrIndices;

			// Scaling of the generated tets
			float Scale;
		} tet_to_tri_params = {4*_mesh->nrVolumes(), 0.9f};
		cmd_buffer->handle()->SetComputeRoot32BitConstants(0, 2, &tet_to_tri_params, 0);
		cmd_buffer->dispatch(4 * _mesh->nrVolumes(), 1, 1);


		cmd_buffer->bindPipeline(_tetToTriMeshCtx.RenderTriMeshPS.get());
		cmd_buffer->handle()->SetGraphicsRootSignature(_tetToTriMeshCtx.TableLayoutGraphics->rootSignature());

		cmd_buffer->handle()->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		cmd_buffer->bindIndexBuffer(_triMeshIndices.get());
		cmd_buffer->bindVertexBuffer(_triMeshVertices.get());
		
		Eigen::Affine3f rot{ Eigen::AngleAxisf{ _tetMeshRotation, Eigen::Vector3f::UnitY() } };
		Eigen::Affine3f trans{ Eigen::Translation3f{ -0.5f, -0.5f, -0.5f } };
		Eigen::Matrix4f mvp = _camera.projection() * _camera.view() * rot.matrix() * trans.matrix();
		cmd_buffer->handle()->SetGraphicsRoot32BitConstants(0, 16, mvp.data(), 0);
		
		cmd_buffer->drawIndexed(3 * 4 * _mesh->nrVolumes(), 1, 0, 0, 0);
	}

	void renderFrameDirectTetMesh(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer)
	{
		using Vcl::Graphics::Runtime::D3D12::PipelineBindPoint;

		cmd_buffer->bindPipeline(_directTetMeshCtx.RenderTetMeshPS.get());
		cmd_buffer->bindDescriptorTable(PipelineBindPoint::Graphics, _directTetMeshCtx.TableGraphics.get());

		cmd_buffer->handle()->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
		cmd_buffer->bindVertexBuffer(_tetMeshIndices.get());

		Eigen::Affine3f rot{ Eigen::AngleAxisf{ _tetMeshRotation, Eigen::Vector3f::UnitY() } };
		Eigen::Affine3f trans{ Eigen::Translation3f{ -0.5f, -0.5f, -0.5f } };
		Eigen::Matrix4f mvp = _camera.projection() * _camera.view() * rot.matrix() * trans.matrix();
		cmd_buffer->handle()->SetGraphicsRoot32BitConstants(0, 16, mvp.data(), 0);

		cmd_buffer->draw(_mesh->nrVolumes(), 1, 0, 0);
	}

	void createDeviceObjects() override
	{
		Application::createDeviceObjects();

		const auto size = swapChain()->bufferSize();
		_camera.setViewport(0.5f * size.first, size.second);
		_camera.setFieldOfView(0.5f * (float) size.first / (float) size.second);
	}

	void updateFrame() override
	{
		_tetMeshRotation += 0.01f;
		const float two_pi = 2 * Vcl::Mathematics::pi<float>();
		if (_tetMeshRotation > two_pi)
			_tetMeshRotation -= two_pi;
	}

	void renderFrame(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv) override
	{
		const auto size = swapChain()->bufferSize();
		const auto w = size.first / 2;
		const auto h = size.second;

		cmd_buffer->handle()->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
		D3D12_VIEWPORT vp{ 0, 0, w, h, 0, 1 };
		cmd_buffer->handle()->RSSetViewports(1, &vp);
		D3D12_RECT sr{ 0, 0, w, h };
		cmd_buffer->handle()->RSSetScissorRects(1, &sr);

		renderFrameTetToTriMesh(cmd_buffer);

		cmd_buffer->handle()->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
		D3D12_VIEWPORT vp2{ w, 0, w, h, 0, 1 };
		cmd_buffer->handle()->RSSetViewports(1, &vp2);
		D3D12_RECT sr2{ w, 0, w+w, h };
		cmd_buffer->handle()->RSSetScissorRects(1, &sr2);

		renderFrameDirectTetMesh(cmd_buffer);
	}

	TetToTriMeshContext _tetToTriMeshCtx;
	DirectTetMeshContext _directTetMeshCtx;

	std::unique_ptr<Vcl::Geometry::TetraMesh> _mesh;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Buffer> _tetMeshIndices;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Buffer> _tetMeshVertices;

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Buffer> _triMeshIndices;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Buffer> _triMeshVertices;

	Vcl::Graphics::Camera _camera{ std::make_shared<Vcl::Graphics::Direct3D::MatrixFactory>() };

	float _tetMeshRotation{ 0 };
};

int main(int argc, char** argv)
{
	DrawTetMeshApplication app;
	return app.run();
}
