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

// Include the relevant parts from the library
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/meshlets.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/runtime/webgpu/resource/buffer.h>
#include <vcl/graphics/runtime/webgpu/resource/shader.h>
#include <vcl/graphics/runtime/webgpu/state/pipelinestate.h>
#include <vcl/graphics/camera.h>
#include <vcl/graphics/surfaceformat.h>
#include <vcl/graphics/trackballcameracontroller.h>
#include <vcl/math/math.h>

#include "mesh.vert.spv.h"
#include "mesh.frag.spv.h"

class DrawMeshApplication final : public Application
{
public:
	DrawMeshApplication()
	: Application("DrawMesh")
	{
		using Vcl::Graphics::Camera;
		using Vcl::Graphics::SurfaceFormat;
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferInitData;
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

		WGPUQueue queue = wgpuDeviceGetQueue(device());

		_vs = std::make_unique<Shader>(device(), ShaderType::VertexShader, 0, MeshSpirvVS);
		_fs = std::make_unique<Shader>(device(), ShaderType::FragmentShader, 0, MeshSpirvFS);

		InputLayoutDescription input_layout{
			{
				{ 0, sizeof(Eigen::Vector3f), VertexDataClassification::VertexDataPerObject },
				{ 1, sizeof(int), VertexDataClassification::VertexDataPerObject },
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
		_gps = std::make_unique<GraphicsPipelineState>(device(), psd, rtd);

		BufferDescription consts_desc = {
			2 * sizeof(Eigen::Matrix4f),
			BufferUsage::CopyDst | BufferUsage::Uniform
		};
		_constants = std::make_unique<Buffer>(device(), consts_desc, nullptr, queue);

		WGPUBindGroupLayout bg_layout;
		bg_layout = wgpuRenderPipelineGetBindGroupLayout(_gps->handle(), 0);

		WGPUBindGroupEntry common_bg_entries[] = {
			{ nullptr, 0, _constants->handle(), 0, _constants->sizeInBytes(), 0, 0 }
		};

		WGPUBindGroupDescriptor common_bg_descriptor = {};
		common_bg_descriptor.layout = bg_layout;
		common_bg_descriptor.entryCount = sizeof(common_bg_entries) / sizeof(WGPUBindGroupEntry);
		common_bg_descriptor.entries = common_bg_entries;
		_commonBindGroup = wgpuDeviceCreateBindGroup(device(), &common_bg_descriptor);

		// Initialize content
		_camera = std::make_unique<Camera>(std::make_shared<Vcl::Graphics::Direct3D::MatrixFactory>());

		_cameraController = std::make_unique<Vcl::Graphics::TrackballCameraController>();
		_cameraController->setCamera(_camera.get());

		updateMesh(queue);
	}

private:
	void updateMesh(WGPUQueue queue)
	{
		using namespace Vcl::Geometry;

		std::unique_ptr<TriMesh> mesh;
		switch (_meshSelection)
		{
		case 0:
		{
			_camera->encloseInFrustum({ 0, 0, 0 }, { 0, 1, -1 }, 1.0f, { 0, 1, 0 });
			mesh = TriMeshFactory::createCube(50, 50, 50);
			break;
		}
		case 1:
		{
			_camera->encloseInFrustum({ 0, 0, 0 }, { 0, 1, -1 }, 5.0f, { 0, 1, 0 });
			mesh = TriMeshFactory::createTorus(4, 1, 50, 50);
			break;
		}
		}
		const auto points = stdext::make_span(mesh->vertices()->data()->data(), mesh->nrVertices() * 3);
		const auto indices = stdext::make_span(reinterpret_cast<int*>(mesh->faces()->data()->data()), mesh->nrFaces() * 3);

		if (_meshletGenerator > 0)
		{
			createMeshlet(queue, static_cast<MeshletGenerator>(_meshletGenerator - 1), points, indices);
		} else
		{
			std::vector<int> colours(mesh->nrVertices(), 0);
			allocateMeshBuffers(queue, stdext::make_span(points), stdext::make_span(colours), stdext::make_span(indices));
		}
	}

	void createMeshlet(WGPUQueue queue, Vcl::Geometry::MeshletGenerator gen, stdext::span<float> points, stdext::span<int> indices)
	{
		using namespace Vcl::Geometry;
		const auto points_in = stdext::make_span(reinterpret_cast<Eigen::Vector3f*>(points.data()), points.size() / 3);
		const auto indices_in = stdext::make_span(reinterpret_cast<uint32_t*>(indices.data()), indices.size());
		const auto meshlets = generateMeshlets(gen, points_in, indices_in);

		// Expand meshlets to triangle list
		std::vector<float> exp_points;
		std::vector<int> exp_colours;
		std::vector<int> exp_indices;

		exp_points.reserve(meshlets.VertexIndices.size() * 3);
		exp_colours.reserve(meshlets.VertexIndices.size());
		exp_indices.reserve(meshlets.Primitives.size());

		for (uint32_t meshletidx = 0; meshletidx < meshlets.Meshlets.size(); meshletidx++)
		{
			const auto& meshlet = meshlets.Meshlets[meshletidx];
			for (uint32_t vidx = meshlet.VertexOffset; vidx < meshlet.VertexOffset + meshlet.VertexCount; vidx++)
			{
				const auto& p = points_in[meshlets.VertexIndices[vidx]];
				exp_points.emplace_back(p.x());
				exp_points.emplace_back(p.y());
				exp_points.emplace_back(p.z());
				exp_colours.emplace_back(meshletidx);
			}
			for (uint32_t pidx = meshlet.PrimitiveOffset; pidx < meshlet.PrimitiveOffset + 3 * meshlet.PrimitiveCount; pidx++)
			{
				exp_indices.emplace_back(meshlet.VertexOffset + meshlets.Primitives[pidx]);
			}
		}

		allocateMeshBuffers(queue, stdext::make_span(exp_points), stdext::make_span(exp_colours), stdext::make_span(exp_indices));
	}

	void allocateMeshBuffers(WGPUQueue queue, stdext::span<float> points, stdext::span<int> colours, stdext::span<int> indices)
	{
		using Vcl::Graphics::Runtime::BufferDescription;
		using Vcl::Graphics::Runtime::BufferInitData;
		using Vcl::Graphics::Runtime::BufferUsage;
		using Vcl::Graphics::Runtime::WebGPU::Buffer;

		BufferDescription vbo_pos_desc = {
			points.size() * sizeof(float),
			BufferUsage::CopyDst | BufferUsage::Vertex
		};
		_vboPos = std::make_unique<Buffer>(device(), vbo_pos_desc, points, queue);

		BufferDescription vbo_col_desc = {
			colours.size() * sizeof(int),
			BufferUsage::CopyDst | BufferUsage::Vertex
		};
		_vboCol = std::make_unique<Buffer>(device(), vbo_col_desc, colours, queue);

		BufferDescription ibo_desc = {
			indices.size() * sizeof(int),
			BufferUsage::CopyDst | BufferUsage::Index
		};
		_ibo = std::make_unique<Buffer>(device(), ibo_desc, indices, queue);
		_nrIndices = indices.size();
	}

	void createDeviceObjects() override
	{
		Application::createDeviceObjects();

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

	void renderFrame(WGPUTextureView back_buffer, WGPUTextureView depth_buffer) override
	{
		using namespace Vcl::Graphics::Runtime;

		WGPUQueue queue = wgpuDeviceGetQueue(device());

		const auto size = _swapChainSize;
		const auto x = 0;
		const auto y = 0;
		const auto w = size.first;
		const auto h = size.second;

		std::array<WGPURenderPassColorAttachment, 1> color_attachments = {};
		color_attachments[0].loadOp = WGPULoadOp_Clear;
		color_attachments[0].storeOp = WGPUStoreOp_Store;
		color_attachments[0].clearColor = { 0.0f, 0.0f, 0.0f, 0.0f };
		color_attachments[0].view = back_buffer;
		WGPURenderPassDepthStencilAttachment depth_attachment = {};
		depth_attachment.clearDepth = 1.0f;
		depth_attachment.clearStencil = 0;
		depth_attachment.depthLoadOp = WGPULoadOp_Clear;
		depth_attachment.depthStoreOp = WGPUStoreOp_Store;
		depth_attachment.stencilLoadOp = WGPULoadOp_Undefined;
		depth_attachment.stencilStoreOp = WGPUStoreOp_Undefined;
		depth_attachment.view = depth_buffer;
		WGPURenderPassDescriptor render_pass_desc = {};
		render_pass_desc.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size());
		render_pass_desc.colorAttachments = color_attachments.data();
		render_pass_desc.depthStencilAttachment = &depth_attachment;

		Eigen::Affine3f rot{ Eigen::AngleAxisf{ _meshRotation, Eigen::Vector3f::UnitY() } };
		struct
		{
			Eigen::Matrix4f model_view;
			Eigen::Matrix4f projection;
		} transform;
		transform.model_view = _camera->view() * rot.matrix();
		transform.projection = _camera->projection();
		wgpuQueueWriteBuffer(queue, _constants->handle(), 0, &transform, sizeof(transform));

		WGPUCommandEncoderDescriptor enc_desc = {};
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device(), &enc_desc);
		{
			WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
			wgpuRenderPassEncoderSetViewport(pass, x, y, w, h, 0, 1);
			wgpuRenderPassEncoderSetScissorRect(pass, x, y, w, h);
			wgpuRenderPassEncoderSetVertexBuffer(pass, 0, _vboPos->handle(), 0, _vboPos->sizeInBytes());
			wgpuRenderPassEncoderSetVertexBuffer(pass, 1, _vboCol->handle(), 0, _vboCol->sizeInBytes());
			wgpuRenderPassEncoderSetIndexBuffer(pass, _ibo->handle(), WGPUIndexFormat_Uint32, 0, _ibo->sizeInBytes());
			wgpuRenderPassEncoderSetPipeline(pass, _gps->handle());
			wgpuRenderPassEncoderSetBindGroup(pass, 0, _commonBindGroup, 0, nullptr);
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
	WGPUBindGroup _commonBindGroup;

	std::unique_ptr<Vcl::Graphics::TrackballCameraController> _cameraController;
	std::unique_ptr<Vcl::Graphics::Camera> _camera;

	int _meshSelection{ 1 };
	int _meshletGenerator{ 2 };
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
