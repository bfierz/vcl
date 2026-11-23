/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2022 Basil Fierz
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

// C++ standard library
#include <array>
#include <optional>
#include <vector>

// Include the relevant parts from the library
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/meshlets.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/runtime/webgpu/resource/buffer.h>
#include <vcl/graphics/runtime/webgpu/resource/shader.h>
#include <vcl/graphics/runtime/webgpu/state/pipelinestate.h>
#include <vcl/graphics/camera.h>
#include <vcl/graphics/surfaceformat.h>
#include <vcl/math/math.h>

#include "mesh.vert.spv.h"
#include "mesh.frag.spv.h"

//! Renders a torus, colouring each meshlet differently
class DrawMeshApplication final : public Application
{
public:
	DrawMeshApplication()
	: Application("DrawMesh")
	{
		using Vcl::Graphics::Camera;
		using Vcl::Graphics::SurfaceFormat;
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferUsage;
		using Vcl::Graphics::Runtime::InputLayoutDescription;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::PrimitiveType;
		using Vcl::Graphics::Runtime::RenderTargetLayout;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::VertexDataClassification;
		using Vcl::Graphics::Runtime::WebGPU::Buffer;
		using Vcl::Graphics::Runtime::WebGPU::GraphicsPipelineState;
		using Vcl::Graphics::Runtime::WebGPU::Shader;

		_vs = std::make_unique<Shader>(_wgpuDevice, ShaderType::VertexShader, 0, MeshSpirvVS);
		_fs = std::make_unique<Shader>(_wgpuDevice, ShaderType::FragmentShader, 0, MeshSpirvFS);

		InputLayoutDescription input_layout{
			{
				{ 0, sizeof(Eigen::Vector3f), VertexDataClassification::VertexDataPerObject },
				{ 1, sizeof(uint32_t), VertexDataClassification::VertexDataPerObject },
			},
			{
				{ "Position", SurfaceFormat::R32G32B32_FLOAT, 0, 0, 0 },
				{ "ColorIdx", SurfaceFormat::R32_UINT, 0, 1, 0 },
			}
		};

		PipelineStateDescription psd;
		psd.VertexShader = _vs.get();
		psd.FragmentShader = _fs.get();
		psd.InputAssembly.Topology = PrimitiveType::Trianglelist;
		psd.InputLayout = input_layout;
		RenderTargetLayout rtd = {};
		rtd.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
		rtd.DepthStencilFormat = SurfaceFormat::D32_FLOAT;
		psd.RenderTargets = rtd;
		_gps = std::make_unique<GraphicsPipelineState>(_wgpuDevice, psd);

		BufferDescription consts_desc = {
			sizeof(Transform),
			BufferUsage::CopyDst | BufferUsage::Uniform
		};
		_constants = std::make_unique<Buffer>(_wgpuDevice, consts_desc);

		WGPUBindGroupEntry transform_entry = {};
		transform_entry.binding = 0;
		transform_entry.buffer = _constants->handle();
		transform_entry.size = _constants->sizeInBytes();

		WGPUBindGroupDescriptor bind_group_desc = {};
		bind_group_desc.layout = wgpuRenderPipelineGetBindGroupLayout(_gps->handle(), 0);
		bind_group_desc.entryCount = 1;
		bind_group_desc.entries = &transform_entry;
		_transformBindGroup = wgpuDeviceCreateBindGroup(_wgpuDevice, &bind_group_desc);
		wgpuBindGroupLayoutRelease(bind_group_desc.layout);

		_camera = std::make_unique<Camera>(std::make_shared<Vcl::Graphics::Direct3D::MatrixFactory>());
		_camera->encloseInFrustum({ 0, 0, 0 }, { 0, 1, -1 }, 5.0f, { 0, 1, 0 });

		createMesh();
	}

	~DrawMeshApplication() override
	{
		releaseDepthBuffer();
		wgpuBindGroupRelease(_transformBindGroup);
	}

private:
	struct Transform
	{
		Eigen::Matrix4f ModelView;
		Eigen::Matrix4f Projection;
	};

	void createMesh()
	{
		using namespace Vcl::Geometry;

		const auto mesh = TriMeshFactory::createTorus(4, 1, 50, 50);

		std::vector<Eigen::Vector3f> points(mesh->nrVertices());
		for (size_t v = 0; v < points.size(); v++)
			points[v] = mesh->vertices()[v];

		std::vector<uint32_t> indices;
		indices.reserve(3 * mesh->nrFaces());
		for (size_t f = 0; f < mesh->nrFaces(); f++)
			for (const auto idx : mesh->faces()[f])
				indices.push_back(idx.id());

		if (_meshletGenerator)
		{
			createMeshlets(*_meshletGenerator, points, indices);
		} else
		{
			const std::vector<uint32_t> colours(points.size(), 0);
			allocateMeshBuffers(points, colours, indices);
		}
	}

	//! Expand the meshlets to a triangle list with one colour per meshlet
	void createMeshlets(Vcl::Geometry::MeshletGenerator generator, const std::vector<Eigen::Vector3f>& points, const std::vector<uint32_t>& indices)
	{
		using namespace Vcl::Geometry;

		const auto meshlets = generateMeshlets(generator, points, indices);

		std::vector<Eigen::Vector3f> meshlet_points;
		std::vector<uint32_t> meshlet_colours;
		std::vector<uint32_t> meshlet_indices;
		meshlet_points.reserve(meshlets.VertexIndices.size());
		meshlet_colours.reserve(meshlets.VertexIndices.size());
		meshlet_indices.reserve(meshlets.Primitives.size());

		for (uint32_t meshlet_idx = 0; meshlet_idx < meshlets.Meshlets.size(); meshlet_idx++)
		{
			const auto& meshlet = meshlets.Meshlets[meshlet_idx];
			for (uint32_t v = 0; v < meshlet.VertexCount; v++)
			{
				meshlet_points.push_back(points[meshlets.VertexIndices[meshlet.VertexOffset + v]]);
				meshlet_colours.push_back(meshlet_idx);
			}
			for (uint32_t p = 0; p < 3 * meshlet.PrimitiveCount; p++)
			{
				meshlet_indices.push_back(meshlet.VertexOffset + meshlets.Primitives[meshlet.PrimitiveOffset + p]);
			}
		}

		allocateMeshBuffers(meshlet_points, meshlet_colours, meshlet_indices);
	}

	void allocateMeshBuffers(const std::vector<Eigen::Vector3f>& points, const std::vector<uint32_t>& colours, const std::vector<uint32_t>& indices)
	{
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferInitData;
		using Vcl::Graphics::Runtime::BufferUsage;
		using Vcl::Graphics::Runtime::WebGPU::Buffer;

		WGPUQueue queue = wgpuDeviceGetQueue(_wgpuDevice);
		const auto create = [this, queue](const auto& data, Vcl::Flags<BufferUsage> usage) {
			const auto size_in_bytes = static_cast<uint32_t>(data.size() * sizeof(data[0]));
			const BufferDescription desc = { size_in_bytes, BufferUsage::CopyDst | usage };
			const BufferInitData init_data = { data.data(), size_in_bytes };
			return std::make_unique<Buffer>(_wgpuDevice, desc, &init_data, queue);
		};

		_vboPos = create(points, BufferUsage::Vertex);
		_vboCol = create(colours, BufferUsage::Vertex);
		_ibo = create(indices, BufferUsage::Index);
		_nrIndices = static_cast<uint32_t>(indices.size());
	}

	void releaseDepthBuffer()
	{
		if (_depthBufferView)
			wgpuTextureViewRelease(_depthBufferView);
		if (_depthBuffer)
			wgpuTextureRelease(_depthBuffer);
		_depthBufferView = nullptr;
		_depthBuffer = nullptr;
	}

	void invalidateDeviceObjects() override
	{
		releaseDepthBuffer();
		Application::invalidateDeviceObjects();
	}

	void createDeviceObjects() override
	{
		Application::createDeviceObjects();

		WGPUTextureDescriptor depth_desc = {};
		depth_desc.usage = WGPUTextureUsage_RenderAttachment;
		depth_desc.dimension = WGPUTextureDimension_2D;
		depth_desc.size = { _swapChainSize.first, _swapChainSize.second, 1 };
		depth_desc.format = WGPUTextureFormat_Depth32Float;
		depth_desc.mipLevelCount = 1;
		depth_desc.sampleCount = 1;
		_depthBuffer = wgpuDeviceCreateTexture(_wgpuDevice, &depth_desc);
		_depthBufferView = wgpuTextureCreateView(_depthBuffer, nullptr);

		_camera->setViewport(_swapChainSize.first, _swapChainSize.second);
		_camera->setFieldOfView((float)_swapChainSize.first / (float)_swapChainSize.second);
	}

	void updateFrame() override
	{
		_meshRotation += 0.01f;
		const float two_pi = 2 * Vcl::Mathematics::pi<float>();
		if (_meshRotation > two_pi)
			_meshRotation -= two_pi;
	}

	void renderFrame(WGPUTextureView back_buffer) override
	{
		WGPUQueue queue = wgpuDeviceGetQueue(_wgpuDevice);

		const Eigen::Affine3f rotation{ Eigen::AngleAxisf{ _meshRotation, Eigen::Vector3f::UnitY() } };
		const Transform transform = { _camera->view() * rotation.matrix(), _camera->projection() };
		wgpuQueueWriteBuffer(queue, _constants->handle(), 0, &transform, sizeof(transform));

		std::array<WGPURenderPassColorAttachment, 1> color_attachments = {};
		color_attachments[0].loadOp = WGPULoadOp_Clear;
		color_attachments[0].storeOp = WGPUStoreOp_Store;
		color_attachments[0].clearValue = { 0.0f, 0.0f, 0.0f, 0.0f };
		color_attachments[0].view = back_buffer;
		color_attachments[0].depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
		WGPURenderPassDepthStencilAttachment depth_attachment = {};
		depth_attachment.view = _depthBufferView;
		depth_attachment.depthLoadOp = WGPULoadOp_Clear;
		depth_attachment.depthStoreOp = WGPUStoreOp_Store;
		depth_attachment.depthClearValue = 1.0f;
		WGPURenderPassDescriptor render_pass_desc = {};
		render_pass_desc.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size());
		render_pass_desc.colorAttachments = color_attachments.data();
		render_pass_desc.depthStencilAttachment = &depth_attachment;

		const auto w = _swapChainSize.first;
		const auto h = _swapChainSize.second;

		WGPUCommandEncoderDescriptor enc_desc = {};
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_wgpuDevice, &enc_desc);
		{
			WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
			wgpuRenderPassEncoderSetViewport(pass, 0, 0, w, h, 0, 1);
			wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);
			wgpuRenderPassEncoderSetPipeline(pass, _gps->handle());
			wgpuRenderPassEncoderSetBindGroup(pass, 0, _transformBindGroup, 0, nullptr);
			wgpuRenderPassEncoderSetVertexBuffer(pass, 0, _vboPos->handle(), 0, _vboPos->sizeInBytes());
			wgpuRenderPassEncoderSetVertexBuffer(pass, 1, _vboCol->handle(), 0, _vboCol->sizeInBytes());
			wgpuRenderPassEncoderSetIndexBuffer(pass, _ibo->handle(), WGPUIndexFormat_Uint32, 0, _ibo->sizeInBytes());
			wgpuRenderPassEncoderDrawIndexed(pass, _nrIndices, 1, 0, 0, 0);
			wgpuRenderPassEncoderEnd(pass);
			wgpuRenderPassEncoderRelease(pass);
		}

		WGPUCommandBufferDescriptor cmd_buffer_desc = {};
		WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
		wgpuCommandEncoderRelease(encoder);
		wgpuQueueSubmit(queue, 1, &cmd_buffer);
		wgpuCommandBufferRelease(cmd_buffer);
	}

	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::Shader> _vs;
	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::Shader> _fs;

	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::GraphicsPipelineState> _gps;
	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::Buffer> _constants;
	WGPUBindGroup _transformBindGroup{ nullptr };

	//! Depth buffer, recreated with the swap-chain
	WGPUTexture _depthBuffer{ nullptr };
	WGPUTextureView _depthBufferView{ nullptr };

	std::unique_ptr<Vcl::Graphics::Camera> _camera;

	//! Meshlet partitioning to visualise; the plain mesh without one
	std::optional<Vcl::Geometry::MeshletGenerator> _meshletGenerator{ Vcl::Geometry::MeshletGenerator::SpatialClustering };
	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::Buffer> _vboPos;
	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::Buffer> _vboCol;
	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::Buffer> _ibo;
	uint32_t _nrIndices{ 0 };
	float _meshRotation{ 0 };
};

// Declare application as global object instead of stack object in main
// in order to prevent it to be cleaned up,
// when 'emscripten_set_main_loop' exists.
DrawMeshApplication app;

int main(int argc, char** argv)
{
	return app.run();
}
