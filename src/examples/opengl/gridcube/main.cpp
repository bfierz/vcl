/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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

#include "shaders/boundinggrid.h"
#include "boundinggrid.vert.spv.h"
#include "boundinggrid.geom.spv.h"
#include "boundinggrid.frag.spv.h"

#include "../common/imguiapp.h"

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

class DynamicBoundingGridExample final : public ImGuiApplication
{
public:
	DynamicBoundingGridExample()
	: ImGuiApplication("Grid Cube")
	{

		using Vcl::Graphics::Runtime::OpenGL::PipelineState;
		using Vcl::Graphics::Runtime::OpenGL::Shader;
		using Vcl::Graphics::Runtime::OpenGL::ShaderProgramDescription;
		using Vcl::Graphics::Runtime::OpenGL::ShaderProgram;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Camera;
		using Vcl::Graphics::SurfaceFormat;

		// Initialize the graphics engine
		Vcl::Graphics::OpenGL::Context::initExtensions();
		Vcl::Graphics::OpenGL::Context::setupDebugMessaging();
		_engine = std::make_unique<Vcl::Graphics::Runtime::OpenGL::GraphicsEngine>();

		// Check availability of features
		if (!Shader::isSpirvSupported())
			throw std::runtime_error("SPIR-V is not supported.");

		// Initialize content
		_camera = std::make_unique<Camera>(std::make_shared<Vcl::Graphics::OpenGL::MatrixFactory>());
		_camera->encloseInFrustum({ 0, 0, 0 }, { 0, -1, 0 }, 20.0f, { 0, 0, 1 });

		_cameraController = std::make_unique<Vcl::Graphics::TrackballCameraController>();
		_cameraController->setCamera(_camera.get());

		Shader boxVert{ ShaderType::VertexShader,   0, BoundingGridVert };
		Shader boxGeom{ ShaderType::GeometryShader, 0, BoundingGridGeom };
		Shader boxFrag{ ShaderType::FragmentShader, 0, BoundingGridFrag };
		PipelineStateDescription boxPSDesc;
		boxPSDesc.VertexShader = &boxVert;
		boxPSDesc.GeometryShader = &boxGeom;
		boxPSDesc.FragmentShader = &boxFrag;
		_boxPipelineState = std::make_unique<PipelineState>(boxPSDesc);

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
		ImGui::Begin("Grid parameters");
		InputUInt("Resolution", &_gridResolution, 1, 10, ImGuiInputTextFlags_None);
		ImGui::End();

		// Update camera
		ImGuiIO& io = ImGui::GetIO();
		const auto size = std::make_pair(1280, 720);
		const auto x = io.MousePos.x;
		const auto y = io.MousePos.y;
		const auto w = size.first;
		const auto h = size.second;
		if (io.MouseClicked[0] && !io.WantCaptureMouse)
		{
			_cameraController->startRotate((float)x / (float)w, (float)y / (float)h);
		}
		else if (io.MouseDown[0])
		{
			_cameraController->rotate((float)x / (float)w, (float)y / (float)h);
		}
		else if (io.MouseReleased[0])
		{
			_cameraController->endRotate();
		}
	}

	void renderFrame() override
	{
		_engine->beginFrame();

		_engine->clear(0, Eigen::Vector4f{0.0f, 0.0f, 0.0f, 1.0f});
		_engine->clear(1.0f);

		Eigen::Matrix4f vp = _camera->projection() * _camera->view();
		Eigen::Matrix4f m = _cameraController->currObjectTransformation();
		Eigen::AlignedBox3f bb{ Eigen::Vector3f{-10.0f, -10.0f, -10.0f }, Eigen::Vector3f{ 10.0f, 10.0f, 10.0f} };
		renderBoundingBox(_engine.get(), bb, _gridResolution, _boxPipelineState, m, vp);

		ImGuiApplication::renderFrame();

		_engine->endFrame();
	}

private:
	void renderBoundingBox
	(
		Vcl::Graphics::Runtime::GraphicsEngine* cmd_queue,
		const Eigen::AlignedBox3f& bb,
		unsigned int resolution,
		Vcl::ref_ptr<Vcl::Graphics::Runtime::PipelineState> ps,
		const Eigen::Matrix4f& M,
		const Eigen::Matrix4f& VP
	)
	{
		// Configure the layout
		cmd_queue->setPipelineState(ps);

		// View on the scene
		auto cbuf_transform = cmd_queue->requestPerFrameConstantBuffer<TransformData>();
		cbuf_transform->ModelMatrix = M;
		cbuf_transform->ViewProjectionMatrix = VP;

		cmd_queue->setConstantBuffer(0, std::move(cbuf_transform));

		// Compute the grid paramters
		float maxSize = bb.diagonal().maxCoeff();
		Eigen::Vector3f origin = bb.center() - 0.5f * maxSize * Eigen::Vector3f::Ones().eval();

		auto cbuf_config = cmd_queue->requestPerFrameConstantBuffer<BoundingGridConfig>();
		cbuf_config->Axis[0] = { 1, 0, 0 };
		cbuf_config->Axis[1] = { 0, 1, 0 };
		cbuf_config->Axis[2] = { 0, 0, 1 };
		cbuf_config->Colours[0] = { 1, 0, 0 };
		cbuf_config->Colours[1] = { 0, 1, 0 };
		cbuf_config->Colours[2] = { 0, 0, 1 };
		cbuf_config->Origin = origin;
		cbuf_config->StepSize = maxSize / (float)resolution;
		cbuf_config->Resolution = (float)resolution;

		cmd_queue->setConstantBuffer(1, std::move(cbuf_config));

		// Render the grid
		// 3 Line-loops with 4 points, N+1 replications of the loops (N tiles)
		cmd_queue->setPrimitiveType(Vcl::Graphics::Runtime::PrimitiveType::LinelistAdj);
		cmd_queue->draw(12, 0, resolution + 1, 0);
	}

private:
	std::unique_ptr<Vcl::Graphics::Runtime::GraphicsEngine> _engine;

private:
	std::unique_ptr<Vcl::Graphics::TrackballCameraController> _cameraController;

private:
	std::unique_ptr<Vcl::Graphics::Camera> _camera;
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _boxPipelineState;

private:
	unsigned int _gridResolution{ 10 };
};

int main(int argc, char** argv)
{
	DynamicBoundingGridExample app;
	return app.run();

	return 0;
}
