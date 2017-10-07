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
#include <vcl/config/opengl.h>

// C++ standard library
#include <iostream>

// GSL
#include <gsl/gsl>

// NanoGUI
#include <nanogui/label.h>
#include <nanogui/layout.h>
#include <nanogui/screen.h>
#include <nanogui/slider.h>
#include <nanogui/textbox.h>
#include <nanogui/window.h>

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

// Force the use of the NVIDIA GPU in an Optimus system
extern "C"
{
	_declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
}

class DynamicBoundingGridExample : public nanogui::Screen
{
public:
	DynamicBoundingGridExample()
	: nanogui::Screen(Eigen::Vector2i(768, 768), "VCL Dynamic Bounding Grid Example")
	{
		using namespace nanogui;

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

		Window *window = new Window(this, "Grid parameters");
		window->setPosition(Vector2i(15, 15));
		window->setLayout(new GroupLayout());

		Widget *panel = new Widget(window);
		panel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 20));

		new Label(panel, "Resolution", "sans-bold");

		TextBox *textBox = new TextBox(panel);
		textBox->setEditable(true);
		textBox->setValue(std::to_string(_gridResolution));
		textBox->setFormat("[1-9][0-9]*");
		textBox->setFontSize(20);
		textBox->setFixedSize(Vector2i(60, 25));
		textBox->setAlignment(TextBox::Alignment::Right);
		textBox->setCallback([this](const std::string& txt) -> bool
		{
			_gridResolution = std::atoi(txt.c_str());
			return true;
		});
		
		performLayout();

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
	}

public:
	bool mouseButtonEvent(const nanogui::Vector2i &p, int button, bool down, int modifiers) override
	{
		if (Widget::mouseButtonEvent(p, button, down, modifiers))
			return true;

		if (down)
		{
			_cameraController->startRotate((float) p.x() / (float) width(), (float) p.y() / (float) height());
		}
		else
		{
			_cameraController->endRotate();
		}

		return true;
	}

	bool mouseMotionEvent(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int modifiers) override
	{
		if (Widget::mouseMotionEvent(p, rel, button, modifiers))
			return true;

		_cameraController->rotate((float)p.x() / (float)width(), (float)p.y() / (float)height());
		return true;
	}

public:
	void drawContents() override
	{
		_engine->beginFrame();

		_engine->clear(0, Eigen::Vector4f{0.0f, 0.0f, 0.0f, 1.0f});
		_engine->clear(1.0f);

		Eigen::Matrix4f vp = _camera->projection() * _camera->view();
		Eigen::Matrix4f m = _cameraController->currObjectTransformation();
		Eigen::AlignedBox3f bb{ Eigen::Vector3f{-10.0f, -10.0f, -10.0f }, Eigen::Vector3f{ 10.0f, 10.0f, 10.0f} };
		renderBoundingBox(_engine.get(), bb, _gridResolution, _boxPipelineState, m, vp);

		_engine->endFrame();
	}

private:
	void renderBoundingBox
	(
		gsl::not_null<Vcl::Graphics::Runtime::GraphicsEngine*> cmd_queue,
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
		
		cmd_queue->setConstantBuffer(0, cbuf_transform);

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

		cmd_queue->setConstantBuffer(1, cbuf_config);
		
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

int main(int /* argc */, char ** /* argv */)
{
	try
	{
		nanogui::init();

		{
			nanogui::ref<DynamicBoundingGridExample> app = new DynamicBoundingGridExample();
			app->drawAll();
			app->setVisible(true);
			nanogui::mainloop();
		}

		nanogui::shutdown();
	}
	catch (const std::runtime_error &e)
	{
		std::string error_msg = std::string("Caught a fatal error: ") + std::string(e.what());
		std::cerr << error_msg << std::endl;
		return -1;
	}

	return 0;
}
