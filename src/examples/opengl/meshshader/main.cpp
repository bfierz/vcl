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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>
#include <vcl/config/opengl.h>

// C++ standard library
#include <iostream>
#include <vector>

// VCL
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/trimesh.h>

#include <vcl/graphics/opengl/glsl/uniformbuffer.h>
#include <vcl/graphics/opengl/context.h>
#include <vcl/graphics/opengl/gl.h>
#include <vcl/graphics/runtime/opengl/resource/shader.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/graphics/runtime/opengl/graphicsengine.h>
#include <vcl/graphics/camera.h>
#include <vcl/graphics/trackballcameracontroller.h>


#include <vcl/graphics/opengl/glsl/uniformbuffer.h>

UNIFORM_BUFFER(0)
TransformData
{
	// Transform to world space
	mat4 ModelMatrix;

	// Transform from world to normalized device coordinates
	mat4 ViewProjectionMatrix;
};

//#include "shaders/boundinggrid.h"
//#include "boundinggrid.vert.spv.h"
//#include "boundinggrid.geom.spv.h"
//#include "boundinggrid.frag.spv.h"

#include "../common/imguiapp.h"

// Simple mesh shader to render fixed triangle:
// https://www.geeks3d.com/20200519/introduction-to-mesh-shaders-opengl-and-vulkan/
const char* SimpleTriangleMS =
	R"(
#version 450
#extension GL_NV_mesh_shader : require
 
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
 
// Custom vertex output block
layout (location = 0) out PerVertexData
{
  vec4 color;
} v_out[];  // [max_vertices]
 
 
const vec3 vertices[3] = {vec3(-1,-1,0), vec3(0,1,0), vec3(1,-1,0)};
const vec3 colors[3] = {vec3(1.0,0.0,0.0), vec3(0.0,1.0,0.0), vec3(0.0,0.0,1.0)};
 
void main()
{
  // Vertices position
  gl_MeshVerticesNV[0].gl_Position = vec4(vertices[0], 1.0); 
  gl_MeshVerticesNV[1].gl_Position = vec4(vertices[1], 1.0); 
  gl_MeshVerticesNV[2].gl_Position = vec4(vertices[2], 1.0); 
 
  // Vertices color
  v_out[0].color = vec4(colors[0], 1.0);
  v_out[1].color = vec4(colors[1], 1.0);
  v_out[2].color = vec4(colors[2], 1.0);
 
  // Triangle indices
  gl_PrimitiveIndicesNV[0] = 0;
  gl_PrimitiveIndicesNV[1] = 1;
  gl_PrimitiveIndicesNV[2] = 2;
 
  // Number of triangles  
  gl_PrimitiveCountNV = 1;
}
)";

const char* SimpleMeshletMS =
	R"(
#version 450
 
#extension GL_NV_mesh_shader : require
 
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 126) out;
 
//-------------------------------------
// transform_ub: Uniform buffer for transformations
//
layout (std140, binding = 0) uniform uniforms_t
{ 
  mat4 ViewProjectionMatrix;
  mat4 ModelMatrix;
} transform_ub;
 
//-------------------------------------
// vb: storage buffer for vertices.
//
struct s_vertex
{
  vec4 position;
  vec4 color;
};
 
layout (std430, binding = 1) buffer _vertices
{
  s_vertex vertices[];
} vb;
 
//-------------------------------------
// mbuf: storage buffer for meshlets.
//
struct s_meshlet
{
  uint vertices[64];
  uint indices[378]; // up to 126 triangles
  uint vertex_count;
  uint index_count;
};
 
layout (std430, binding = 2) buffer _meshlets
{
  s_meshlet meshlets[];
} mbuf;
 
// Mesh shader output block.
//
layout (location = 0) out PerVertexData
{
  vec4 color;
} v_out[];   // [max_vertices]
 
// Color table for drawing each meshlet with a different color.
//
#define MAX_COLORS 10
vec3 meshletcolors[MAX_COLORS] = {
  vec3(1,0,0), 
  vec3(0,1,0),
  vec3(0,0,1),
  vec3(1,1,0),
  vec3(1,0,1),
  vec3(0,1,1),
  vec3(1,0.5,0),
  vec3(0.5,1,0),
  vec3(0,0.5,1),
  vec3(1,1,1)
  };
 
void main()
{
  uint mi = gl_WorkGroupID.x;
  uint thread_id = gl_LocalInvocationID.x;
 
  uint vertex_count = mbuf.meshlets[mi].vertex_count;
  for (uint i = 0; i < vertex_count; ++i)
  {
    uint vi = mbuf.meshlets[mi].vertices[i];
 
    vec4 Pw = transform_ub.ModelMatrix * vb.vertices[vi].position;
    vec4 P = transform_ub.ViewProjectionMatrix * Pw;
 
    // GL->VK conventions...
    P.y = -P.y; P.z = (P.z + P.w) / 2.0;
 
    gl_MeshVerticesNV[i].gl_Position = P;
 
    v_out[i].color = vb.vertices[vi].color * vec4(meshletcolors[mi%MAX_COLORS], 1.0);
  }
 
  uint index_count = mbuf.meshlets[mi].index_count;
  gl_PrimitiveCountNV = uint(index_count) / 3;
 
  for (uint i = 0; i < index_count; ++i)
  {
    gl_PrimitiveIndicesNV[i] = uint(mbuf.meshlets[mi].indices[i]);
  }
}
)";

const char* SimpleTriangleFS =
	R"(
#version 450
 
layout(location = 0) out vec4 FragColor;
 
in PerVertexData
{
  vec4 color;
} fragIn;  
 
void main()
{
  FragColor = fragIn.color;
}
)";

// Force the use of the NVIDIA GPU in an Optimus system
#ifdef VCL_ABI_WINAPI
extern "C"
{
	_declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
}
#endif

bool InputUInt(const char* label, unsigned int* v, int step, int step_fast, ImGuiInputTextFlags flags)
{
	// Hexadecimal input provided as a convenience but the flag name is awkward. Typically you'd use InputText() to parse your own data, if you want to handle prefixes.
	const char* format = (flags & ImGuiInputTextFlags_CharsHexadecimal) ? "%08X" : "%d";
	return ImGui::InputScalar(label, ImGuiDataType_U32, (void*)v, (void*)(step > 0 ? &step : NULL), (void*)(step_fast > 0 ? &step_fast : NULL), format, flags);
}

class SimpleMeshletMesh
{
public:
	struct Meshlet
	{
		std::array<uint32_t, 60> Vertices; ///< Indices into the vertex data
		std::array<uint32_t, 384> Indices; ///< Primitive indices into the \see Vertices array
		uint32_t VertexCount;
		uint32_t PrimitiveCount;
	};

	SimpleMeshletMesh(const Vcl::Geometry::TriMesh& mesh)
	{
		// Copy position data

		// Create meshlets
		// * Pick triangle
		// * Add vertices to meshlet (if enough space, else start new meshlet)
		// * Append indices
	}

	/// Position data
	std::vector<Eigen::Vector3f> _positions;

	/// Meshlets
	std::vector<Meshlet> _meshlets;
};

// Sample implementation according to https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/
class MeshletMesh
{
public:
	struct MeshletDesc
	{
		uint32_t VertexCount; // number of vertices used
		uint32_t PrimCount;   // number of primitives (triangles) used
		uint32_t VertexBegin; // offset into vertexIndices
		uint32_t PrimBegin;   // offset into primitiveIndices
	};

	std::vector<MeshletDesc> MeshletInfos;
	std::vector<uint8_t> PrimitiveIndices;
	std::vector<uint32_t> VertexIndices;
};

class MeshShaderExample final : public ImGuiApplication
{
public:
	MeshShaderExample()
	: ImGuiApplication("Mesh Shader")
	{

		using Vcl::Graphics::Camera;
		using Vcl::Graphics::SurfaceFormat;
		using Vcl::Graphics::Runtime::GraphicsMeshShaderPipelineStateDescription;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::OpenGL::GraphicsMeshShaderPipelineState;
		using Vcl::Graphics::Runtime::OpenGL::Shader;

		// Initialize the graphics engine
		Vcl::Graphics::OpenGL::Context::initExtensions();
		Vcl::Graphics::OpenGL::Context::setupDebugMessaging();
		_engine = std::make_unique<Vcl::Graphics::Runtime::OpenGL::GraphicsEngine>();

		// Check availability of features
		if (!Shader::areMeshShadersSupported())
			throw std::runtime_error("Mesh shaders are not supported.");
		if (!Shader::isSpirvSupported())
			throw std::runtime_error("SPIR-V is not supported.");

		_maxMeshOutputVertices = Vcl::Graphics::OpenGL::GL::getInteger(GL_MAX_MESH_OUTPUT_VERTICES_NV);
		_maxMeshOutputPrimitives = Vcl::Graphics::OpenGL::GL::getInteger(GL_MAX_MESH_OUTPUT_PRIMITIVES_NV);

		// Initialize content
		_camera = std::make_unique<Camera>(std::make_shared<Vcl::Graphics::OpenGL::MatrixFactory>());
		_camera->encloseInFrustum({ 0, 0, 0 }, { 0, -1, 0 }, 20.0f, { 0, 0, 1 });

		_cameraController = std::make_unique<Vcl::Graphics::TrackballCameraController>();
		_cameraController->setCamera(_camera.get());

		Shader simpleTriangleMS{ ShaderType::MeshShader, 0, SimpleTriangleMS };
		Shader simpleTriangleFS{ ShaderType::FragmentShader, 0, SimpleTriangleFS };
		GraphicsMeshShaderPipelineStateDescription simpleTrianglePSDesc;
		simpleTrianglePSDesc.MeshShader = &simpleTriangleMS;
		simpleTrianglePSDesc.FragmentShader = &simpleTriangleFS;
		simpleTrianglePSDesc.DepthStencil.DepthEnable = false;
		simpleTrianglePSDesc.Rasterizer.CullMode = Vcl::Graphics::Runtime::CullModeMethod::None;
		_simpleTrianglePS = std::make_unique<GraphicsMeshShaderPipelineState>(simpleTrianglePSDesc);

		const auto size = std::make_pair(1280, 720);
		_camera->setViewport(size.first, size.second);
		_camera->setFieldOfView((float)size.first / (float)size.second);
		_camera->encloseInFrustum({ 0, 0, 0 }, { 0, -1, 0 }, 15.0f, { 0, 0, 1 });
	}

public:
	void updateFrame() override
	{
		ImGuiApplication::updateFrame();

		// Update UI
		ImGui::LabelText("Mesh Shader Maximum Output Vertices", "%d", _maxMeshOutputVertices);
		ImGui::LabelText("Mesh Shader Maximum Output Primitives", "%d", _maxMeshOutputPrimitives);
		//ImGui::Begin("Exmpales parameters");
		//InputUInt("Example", &_example, 1, 1, ImGuiInputTextFlags_None);
		//ImGui::End();

		// Update camera
		ImGuiIO& io = ImGui::GetIO();
		const auto size = std::make_pair(1280, 720);
		const auto x = io.MousePos.x;
		const auto y = io.MousePos.y;
		const auto w = size.first;
		const auto h = size.second;
		if (io.MouseClicked[0] && !io.WantCaptureMouse)
			_cameraController->startRotate((float)x / (float)w, (float)y / (float)h);
		else if (io.MouseDown[0])
			_cameraController->rotate((float)x / (float)w, (float)y / (float)h);
		else if (io.MouseReleased[0])
			_cameraController->endRotate();
	}

	void renderFrame() override
	{
		_engine->beginFrame();

		_engine->clear(0, Eigen::Vector4f{ 0.0f, 0.0f, 0.0f, 1.0f });
		_engine->clear(1.0f);

		Eigen::Matrix4f vp = _camera->projection() * _camera->view();
		Eigen::Matrix4f m = _cameraController->currObjectTransformation();
		renderSimpleTriangle(_engine.get(), _simpleTrianglePS, m, vp);

		ImGuiApplication::renderFrame();

		_engine->endFrame();
	}

private:
	void renderSimpleTriangle(
		Vcl::Graphics::Runtime::GraphicsEngine* cmd_queue,
		Vcl::ref_ptr<Vcl::Graphics::Runtime::PipelineState> ps,
		const Eigen::Matrix4f& M,
		const Eigen::Matrix4f& VP)
	{
		cmd_queue->setPipelineState(ps);
		cmd_queue->drawMeshTasks(0, 1);
	}

	void renderMeshlet(
		Vcl::Graphics::Runtime::GraphicsEngine* cmd_queue,
		Vcl::ref_ptr<Vcl::Graphics::Runtime::PipelineState> ps,
		const Eigen::Matrix4f& M,
		const Eigen::Matrix4f& VP)
	{
		// View on the scene
		auto cbuf_transform = cmd_queue->requestPerFrameConstantBuffer<TransformData>();
		cbuf_transform->ModelMatrix = M;
		cbuf_transform->ViewProjectionMatrix = VP;
		cmd_queue->setConstantBuffer(0, std::move(cbuf_transform));



		cmd_queue->setPipelineState(ps);
		cmd_queue->drawMeshTasks(0, 1);
	}

private:
	std::unique_ptr<Vcl::Graphics::Runtime::GraphicsEngine> _engine;

private:
	std::unique_ptr<Vcl::Graphics::TrackballCameraController> _cameraController;

private:
	std::unique_ptr<Vcl::Graphics::Camera> _camera;
	std::unique_ptr<Vcl::Graphics::Runtime::PipelineState> _simpleTrianglePS;

private:
	/// OpenGL device constant: GL_MAX_MESH_OUTPUT_VERTICES_NV
	int _maxMeshOutputVertices{ 0 };
	/// OpenGL device constant: GL_MAX_MESH_OUTPUT_PRIMITIVES_NV
	int _maxMeshOutputPrimitives{ 0 };

	unsigned int _example{ 0 };
};

int main(int argc, char** argv)
{
	MeshShaderExample app;
	return app.run();

	return 0;
}
