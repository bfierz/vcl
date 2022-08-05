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

// VCL
#include <vcl/graphics/opengl/glsl/uniformbuffer.h>
#include <vcl/graphics/opengl/context.h>
#include <vcl/graphics/runtime/opengl/resource/shader.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/graphics/runtime/opengl/graphicsengine.h>
#include <vcl/graphics/camera.h>
#include <vcl/graphics/trackballcameracontroller.h>

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
		// ImGui::Begin("Grid parameters");
		// InputUInt("Resolution", &_gridResolution, 1, 10, ImGuiInputTextFlags_None);
		// ImGui::End();

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

private:
	std::unique_ptr<Vcl::Graphics::Runtime::GraphicsEngine> _engine;

private:
	std::unique_ptr<Vcl::Graphics::TrackballCameraController> _cameraController;

private:
	std::unique_ptr<Vcl::Graphics::Camera> _camera;
	std::unique_ptr<Vcl::Graphics::Runtime::PipelineState> _simpleTrianglePS;

private:
	// unsigned int _gridResolution{ 10 };
};

int main(int argc, char** argv)
{
	MeshShaderExample app;
	return app.run();

	return 0;
}
