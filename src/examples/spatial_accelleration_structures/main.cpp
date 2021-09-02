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

#include <unordered_map>

#include "../opengl/common/imguiapp.h"

#include "ImFileDialog.h"

 // VCL
#include <vcl/core/container/octree.h>
#include <vcl/core/span.h>

#include <vcl/geometry/io/serialiser_obj.h>
#include <vcl/geometry/io/trimesh_serialiser.h>
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/meshoperations.h>

#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/resource/shader.h>
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/graphics/camera.h>
#include <vcl/graphics/matrixfactory.h>
#include <vcl/graphics/trackballcameracontroller.h>

#include "../opengl/common/shaders/3DSceneBindings.h"
#include "shaders/boundingbox.h"
#include "boundingbox.vert.spv.h"
#include "boundingbox.frag.spv.h"
#include "trimesh.h"
#include "trimesh.vert.spv.h"
#include "trimesh.frag.spv.h"

// Force the use of the NVIDIA GPU in an Optimus system
#ifdef VCL_ABI_WINAPI
extern "C"
{
	_declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
}
#endif

struct MeshDeviceBuffers
{
	void update(const Vcl::Geometry::TriMesh& mesh)
	{
		using namespace Vcl::Geometry;
		using namespace Vcl::Graphics::Runtime;

		NrIndices = static_cast<uint32_t>(mesh.nrFaces() * 3);
		NrVertices = static_cast<uint32_t>(mesh.nrVertices());
		Bounds = computeMeshBoundingBox(mesh);

		// Create the index buffer
		BufferDescription idx_desc;
		idx_desc.Usage = BufferUsage::Index;
		idx_desc.SizeInBytes = static_cast<uint32_t>(mesh.nrFaces() * sizeof(IndexDescriptionTrait<TriMesh>::Face));

		BufferInitData idx_data;
		idx_data.Data = mesh.faces()->data();
		idx_data.SizeInBytes = idx_desc.SizeInBytes;

		Indices = std::make_unique<OpenGL::Buffer>(idx_desc, &idx_data);

		// Create the position buffer
		BufferDescription pos_desc;
		pos_desc.Usage = BufferUsage::Vertex;
		pos_desc.SizeInBytes = static_cast<uint32_t>(mesh.nrVertices() * sizeof(IndexDescriptionTrait<TriMesh>::Vertex));

		BufferInitData pos_data;
		pos_data.Data = mesh.vertices()->data();
		pos_data.SizeInBytes = pos_desc.SizeInBytes;

		Positions = std::make_unique<OpenGL::Buffer>(pos_desc, &pos_data);

		// Create the normal buffer
		if (const auto normals = mesh.vertexProperty<Eigen::Vector3f>("Normals"))
		{
			BufferDescription normal_desc;
			normal_desc.Usage = BufferUsage::Vertex;
			normal_desc.SizeInBytes = static_cast<uint32_t>(mesh.nrVertices() * sizeof(Eigen::Vector3f));

			BufferInitData normal_data;
			normal_data.Data = normals->data();
			normal_data.SizeInBytes = normal_desc.SizeInBytes;

			Normals = std::make_unique<OpenGL::Buffer>(normal_desc, &normal_data);
		}
	}

	//! Number of indices to render
	uint32_t NrIndices{ 0 };

	//! Number of vertices to render
	uint32_t NrVertices{ 0 };

	//! Mesh bounding box
	Eigen::AlignedBox3f Bounds;

	//! Mesh index buffer
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> Indices;
	//! Mesh vertex buffer (position)
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> Positions;
	//! Mesh vertex buffer (normal)
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> Normals;

private:
	Eigen::AlignedBox3f computeMeshBoundingBox(const Vcl::Geometry::TriMesh& mesh)
	{
		const auto& vertices = mesh.vertices();
		stdext::span<Eigen::Vector3f> positions{ vertices->data(), vertices->size() };

		Eigen::AlignedBox3f bb;
		for (const auto& p : positions)
			bb.extend(p);

		return bb;
	}
};

struct AccelerationStructureRenderData
{
	void update(stdext::span<const Eigen::AlignedBox3f> boxes)
	{
		using namespace Vcl::Geometry;
		using namespace Vcl::Graphics::Runtime;

		if (BoundingBoxMesh.NrVertices == 0)
		{
			const auto bb_mesh = TriMeshFactory::createCube(1, 1, 1);
			BoundingBoxMesh.update(*bb_mesh);
		}

		std::vector<Eigen::Vector3f> minima, maxima;
		minima.reserve(boxes.size());
		maxima.reserve(boxes.size());
		std::transform(std::begin(boxes), std::end(boxes), std::back_inserter(minima), [](const auto& box) {return box.min(); });
		std::transform(std::begin(boxes), std::end(boxes), std::back_inserter(maxima), [](const auto& box) {return box.max(); });

		NrBoxes = boxes.size();

		// Create the bb minima buffer
		BufferDescription bb_desc;
		bb_desc.Usage = BufferUsage::Vertex;
		bb_desc.SizeInBytes = static_cast<uint32_t>(boxes.size() * sizeof(Eigen::Vector3f));

		BufferInitData bb_min_data;
		bb_min_data.Data = minima.data();
		bb_min_data.SizeInBytes = bb_desc.SizeInBytes;

		DevBBMin = std::make_unique<OpenGL::Buffer>(bb_desc, &bb_min_data);

		BufferInitData bb_max_data;
		bb_max_data.Data = maxima.data();
		bb_max_data.SizeInBytes = bb_desc.SizeInBytes;

		DevBBMax = std::make_unique<OpenGL::Buffer>(bb_desc, &bb_max_data);
	}

	//! Number of bounding boxes to render
	unsigned int NrBoxes{ 0 };

	//! Bounding box minima
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> DevBBMin;

	//! Bounding box maxima
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> DevBBMax;

	//! Bounding box mesh
	MeshDeviceBuffers BoundingBoxMesh;
};

class DemoImGuiApplication final : public ImGuiApplication
{
public:
	using Texture2D = Vcl::Graphics::Runtime::OpenGL::Texture2D;

	DemoImGuiApplication(const char* title)
		: ImGuiApplication(title)
	{
		// ImFileDialog requires you to set the CreateTexture and DeleteTexture
		ifd::FileDialog::Instance().CreateTexture = [this](uint8_t* data, int w, int h, char fmt) -> void* {

			GLuint tex;
			glGenTextures(1, &tex);
			glBindTexture(GL_TEXTURE_2D, tex);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, (fmt == 0) ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, 0);
			return (void*)tex;
		};
		ifd::FileDialog::Instance().DeleteTexture = [this](void* tex) {
			GLuint tex_id = (GLuint)((uintptr_t)tex);
			glDeleteTextures(1, &tex_id);
		};

		setupPipelines();
	}

private:
	void updateFrame() override
	{
		ImGuiApplication::updateFrame();

		ImGui::Begin("Mesh");
		if (ImGui::Button("Open Mesh File"))
			ifd::FileDialog::Instance().Open("MeshOpenDialog", "Open a mesh", "Mesh file (*.obj){.obj},.*", true);

		// Mesh statistics
		ImGui::Text("Nr Vertices: "); ImGui::SameLine(); ImGui::LabelText("", "%i", _devMesh.NrVertices);
		ImGui::Text("Nr Faces: "); ImGui::SameLine(); ImGui::LabelText("", "%i", _devMesh.NrIndices / 3);

		ImGui::End();

		if (ifd::FileDialog::Instance().IsDone("MeshOpenDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				_mesh.reset();

				const std::vector<std::filesystem::path>& res = ifd::FileDialog::Instance().GetResults();
				for (const auto& r : res)
				{
					using namespace Vcl::Geometry::IO;
					TriMeshDeserialiser reader;
					ObjSerialiser{}.load(&reader, r.string());
					_mesh = reader.fetch();
					break;
				}

				if (_mesh)
				{
					meshToDeviceBuffer();

					const auto bb = _devMesh.Bounds;
					_camera.encloseInFrustum(bb.center(), { 0, 0, 1 }, bb.diagonal().norm() / 4, { 0, 1, 0 });
					_cameraController.setCamera(&_camera);
					_cameraController.setRotationCenter(bb.center());
					_cameraController.setMode(Vcl::Graphics::CameraMode::Object);
				}
			}
			ifd::FileDialog::Instance().Close();
		}

		// Update camera
		const auto size = ImGui::GetMainViewport()->Size;
		if (size.x != _camera.viewportWidth() || size.y != _camera.viewportHeight())
		{
			_camera.setViewport(size.x, size.y);
			_camera.setFieldOfView(size.x / size.y);
		}

		const auto& io = ImGui::GetIO();
		const auto x = io.MousePos.x;
		const auto y = io.MousePos.y;
		const auto w = size.x;
		const auto h = size.y;
		if (io.MouseClicked[0] && !io.WantCaptureMouse)
		{
			_cameraController.startRotate(x / w, y / h);
		}
		else if (io.MouseDown[0])
		{
			_cameraController.rotate(x / w, y / h);
		}
		else if (io.MouseReleased[0])
		{
			_cameraController.endRotate();
		}
	}
	void renderFrame(Vcl::Graphics::Runtime::GraphicsEngine& engine) override
	{
		using namespace Vcl::Graphics::Runtime;

		RenderPassDescription rp_desc = {};
		rp_desc.RenderTargetAttachments.resize(1);
		rp_desc.RenderTargetAttachments[0].View = nullptr;
		rp_desc.RenderTargetAttachments[0].ClearColor = { clear_color.x, clear_color.y, clear_color.z, clear_color.z };
		rp_desc.RenderTargetAttachments[0].LoadOp = AttachmentLoadOp::Clear;
		rp_desc.DepthStencilTargetAttachment.View = nullptr;
		rp_desc.DepthStencilTargetAttachment.ClearDepth = 1.0f;
		rp_desc.DepthStencilTargetAttachment.DepthLoadOp = AttachmentLoadOp::Clear;
		engine.beginRenderPass(rp_desc);

		renderScene(engine);

		engine.endRenderPass();
		
		ImGuiApplication::renderFrame(engine);
	}

	void meshToDeviceBuffer()
	{
		using namespace Vcl::Geometry;
		using namespace Vcl::Graphics::Runtime;

		// Create the normal buffer
		std::vector<Eigen::Vector3f> normals(_mesh->vertices()->size(), Eigen::Vector3f::Zero());
		computeNormals<IndexDescriptionTrait<TriMesh>::VertexId>({ _mesh->faces()->data(), _mesh->faces()->size() }, { _mesh->vertices()->data(), _mesh->vertices()->size() }, { normals.data(), normals.size() });

		auto normals_prop = _mesh->addVertexProperty<Eigen::Vector3f>("Normals", Eigen::Vector3f::Zero());
		for (size_t i = 0; i < normals.size(); i++)
			normals_prop[i] = normals[i].normalized();

		_devMesh.update(*_mesh);

		// Stream bounding boxes
		_octree = std::make_unique<Vcl::Core::Octree>(_devMesh.Bounds, 10);
		std::vector<Eigen::Vector3f> points(_mesh->vertices()->data(), _mesh->vertices()->data() + _mesh->vertices()->size());
		_octree->assign(std::move(points));

		const auto boxes = _octree->collectFullBoundingBoxes();
		_devAccStructRenderData.update(boxes);
	}

	void setupPipelines()
	{
		using namespace Vcl::Graphics::Runtime;
		using namespace Vcl::Graphics;

		InputLayoutDescription tri_mesh_layout =
		{
			{
				{ 0, sizeof(Eigen::Vector3f), VertexDataClassification::VertexDataPerObject },
				{ 1, sizeof(Eigen::Vector3f), VertexDataClassification::VertexDataPerObject }
			},
			{
				{ "Position",  SurfaceFormat::R32G32B32_FLOAT, 0, 0, 0 },
				{ "Normal",  SurfaceFormat::R32G32B32_FLOAT, 0, 1, 0 }
			}
		};

		Runtime::OpenGL::Shader tri_mesh_vert{ ShaderType::VertexShader, 0, TriMeshVert };
		Runtime::OpenGL::Shader tri_mesh_frag{ ShaderType::FragmentShader, 0, TriMeshFrag };

		PipelineStateDescription opaque_trimesh_ps_desc;
		opaque_trimesh_ps_desc.InputLayout = tri_mesh_layout;
		opaque_trimesh_ps_desc.InputAssembly.Topology = PrimitiveType::Trianglelist;
		opaque_trimesh_ps_desc.VertexShader = &tri_mesh_vert;
		opaque_trimesh_ps_desc.FragmentShader = &tri_mesh_frag;
		_opaqueTriMeshPipelineState = Vcl::make_owner<Runtime::OpenGL::PipelineState>(opaque_trimesh_ps_desc);

		InputLayoutDescription bb_mesh_layout =
		{
			{
				{ 0, sizeof(Eigen::Vector3f), VertexDataClassification::VertexDataPerObject },
				{ 1, sizeof(Eigen::Vector3f), VertexDataClassification::VertexDataPerInstance },
				{ 2, sizeof(Eigen::Vector3f), VertexDataClassification::VertexDataPerInstance }
			},
			{
				{ "Position", SurfaceFormat::R32G32B32_FLOAT, 0, 0, 0 },
				{ "Min",      SurfaceFormat::R32G32B32_FLOAT, 0, 1, 0 },
				{ "Max",      SurfaceFormat::R32G32B32_FLOAT, 0, 2, 0 }
			}
		};

		Runtime::OpenGL::Shader bb_mesh_vert{ ShaderType::VertexShader, 0, BoundingBoxVert };
		Runtime::OpenGL::Shader bb_mesh_frag{ ShaderType::FragmentShader, 0, BoundingBoxFrag };

		PipelineStateDescription bb_ps_desc;
		bb_ps_desc.InputLayout = bb_mesh_layout;
		bb_ps_desc.InputAssembly.Topology = PrimitiveType::Trianglelist;
		bb_ps_desc.VertexShader = &bb_mesh_vert;
		bb_ps_desc.FragmentShader = &bb_mesh_frag;
		bb_ps_desc.Blend.RenderTarget[0].BlendEnable = true;
		bb_ps_desc.Blend.RenderTarget[0].BlendOp = BlendOperation::Add;
		bb_ps_desc.Blend.RenderTarget[0].SrcBlend = Blend::SrcAlpha;
		bb_ps_desc.Blend.RenderTarget[0].DestBlend = Blend::One;
		bb_ps_desc.DepthStencil.DepthEnable = false;
		_bbPipelineState = Vcl::make_owner<Runtime::OpenGL::PipelineState>(bb_ps_desc);
	}

	void renderScene(Vcl::Graphics::Runtime::GraphicsEngine& engine)
	{
		using namespace Vcl::Geometry;

		if (!_mesh)
			return;

		const Eigen::Vector4f frustum = { std::tan(_camera.fieldOfView() / 2.0f), (float)_camera.viewportWidth() / (float)_camera.viewportHeight(), _camera.nearPlane(), _camera.farPlane() };
		const Eigen::Matrix4f M = _cameraController.currObjectTransformation();
		const Eigen::Matrix4f V = _camera.view();
		const Eigen::Matrix4f P = _camera.projection();

		auto cbuffer_cam = engine.requestPerFrameConstantBuffer<PerFrameCameraData>();
		cbuffer_cam->Viewport = Eigen::Vector4f{ 0, 0, (float)_camera.viewportWidth(), (float)_camera.viewportHeight() };
		cbuffer_cam->Frustum = frustum;
		cbuffer_cam->ViewMatrix = V;
		cbuffer_cam->ProjectionMatrix = P;
		engine.setConstantBuffer(PER_FRAME_CAMERA_DATA_LOC, std::move(cbuffer_cam));

		{
			engine.setPipelineState(_opaqueTriMeshPipelineState);

			// Bind the buffers
			engine.setIndexBuffer(*_devMesh.Indices);
			engine.setVertexBuffer(0, *_devMesh.Positions, 0, static_cast<int>(sizeof(IndexDescriptionTrait<TriMesh>::Vertex)));
			engine.setVertexBuffer(1, *_devMesh.Normals, 0, static_cast<int>(sizeof(Eigen::Vector3f)));

			// Write local object data
			auto cbuffer_obj = engine.requestPerFrameConstantBuffer<TriMeshPerObjectData>();
			cbuffer_obj->ModelMatrix = M;
			engine.setConstantBuffer(VCL_GLSL_TRIMESH_PER_OBJECT_DATA_LOC, std::move(cbuffer_obj));

			// Render the mesh
			engine.setPrimitiveType(Vcl::Graphics::Runtime::PrimitiveType::Trianglelist);
			engine.drawIndexed(_devMesh.NrIndices);
		}

		{
			engine.setPipelineState(_bbPipelineState);

			// Bind the buffers
			engine.setIndexBuffer(*_devAccStructRenderData.BoundingBoxMesh.Indices);
			engine.setVertexBuffer(0, *_devAccStructRenderData.BoundingBoxMesh.Positions, 0, static_cast<int>(sizeof(IndexDescriptionTrait<TriMesh>::Vertex)));
			engine.setVertexBuffer(1, *_devAccStructRenderData.DevBBMin, 0, static_cast<int>(sizeof(Eigen::Vector3f)));
			engine.setVertexBuffer(2, *_devAccStructRenderData.DevBBMax, 0, static_cast<int>(sizeof(Eigen::Vector3f)));

			// Write local object data
			auto cbuffer_bb = engine.requestPerFrameConstantBuffer<BoundingBoxPerObjectData>();
			cbuffer_bb->ModelMatrix = M;
			cbuffer_bb->Colour = { 0.8f, 0.2f, 0.2f, 0.1f };
			engine.setConstantBuffer(VCL_GLSL_BOUNDINGBOX_PER_OBJECT_DATA_LOC, std::move(cbuffer_bb));

			// Render the mesh
			engine.setPrimitiveType(Vcl::Graphics::Runtime::PrimitiveType::Trianglelist);
			engine.drawIndexed(_devAccStructRenderData.BoundingBoxMesh.NrIndices, 0, _devAccStructRenderData.NrBoxes);
		}
	}

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	//! Current mesh to process
	std::unique_ptr<Vcl::Geometry::TriMesh> _mesh;

	//! Octree with mesh data
	std::unique_ptr<Vcl::Core::Octree> _octree;

	//! Mesh buffers
	MeshDeviceBuffers _devMesh;

	//! Accelleration structure render data
	AccelerationStructureRenderData _devAccStructRenderData;

	//! Scene camera
	Vcl::Graphics::Camera _camera{ std::make_shared<Vcl::Graphics::OpenGL::MatrixFactory>() };

	//! Scene camera controller
	Vcl::Graphics::TrackballCameraController _cameraController;

	//! Opaque TriMesh pipeline state
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _opaqueTriMeshPipelineState;

	//! Opaque BB pipeline state
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _bbPipelineState;
};

int main(int argc, char** argv)
{
	DemoImGuiApplication app{ "Spatial Accelleration Structure Demo" };
	return app.run();
}
