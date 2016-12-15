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
#include <vcl/config/cuda.h>
#include <vcl/config/opengl.h>

// C++ standard library
#include <iostream>

// NanoGUI
#include <nanogui/label.h>
#include <nanogui/layout.h>
#include <nanogui/screen.h>
#include <nanogui/slider.h>
#include <nanogui/textbox.h>
#include <nanogui/window.h>

// Cuda
#include <cudaGL.h>

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/device.h>
#include <vcl/compute/cuda/platform.h>

#include <vcl/graphics/runtime/opengl/resource/shader.h>
#include <vcl/graphics/runtime/opengl/resource/texture3d.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/graphics/runtime/opengl/state/sampler.h>
#include <vcl/graphics/camera.h>
#include <vcl/graphics/trackballcameracontroller.h>

#include <vcl/physics/fluid/cuda/cudacentergrid.h>
#include <vcl/physics/fluid/cuda/cudaeulerfluidsimulation.h>

// Local
#include "shaders/boundinggrid.vert"
#include "shaders/boundinggrid.geom"
#include "shaders/boundinggrid.frag"
#include "shaders/smoke.vert"
#include "shaders/smoke.geom"
#include "shaders/smoke.frag"

// Force the use of the NVIDIA GPU in an Optimus system
extern "C"
{
	_declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
}

class RealTimeWaveletTurbulence : public nanogui::Screen
{
public:
	RealTimeWaveletTurbulence()
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

		// Initialize Cuda
		initializeCuda();

		// Initialize glew
		glewExperimental = GL_TRUE;
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			/* Problem: glewInit failed, something is seriously wrong. */
			std::cout << "Error: GLEW: " << glewGetErrorString(err) << std::endl;
		}

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

		Shader boxVert{ ShaderType::VertexShader,   0, vert_shader };
		Shader boxGeom{ ShaderType::GeometryShader, 0, geom_shader };
		Shader boxFrag{ ShaderType::FragmentShader, 0, frag_shader };
		PipelineStateDescription boxPSDesc;
		boxPSDesc.VertexShader   = &boxVert;
		boxPSDesc.GeometryShader = &boxGeom;
		boxPSDesc.FragmentShader = &boxFrag;
		_boxPipelineState = std::make_unique<PipelineState>(boxPSDesc);

		// Initialize the fluid solver
		_fluidSolver = std::make_unique<Vcl::Physics::Fluid::Cuda::EulerFluidSimulation>(_cudaContext);
		auto cmd_queue = Vcl::static_pointer_cast<Vcl::Compute::Cuda::CommandQueue>(_cudaContext->defaultQueue());

		// Initialize the fluid grid
		int c = 128;
		_fluidGrid = std::make_unique<Vcl::Physics::Fluid::Cuda::CenterGrid>(_cudaContext, cmd_queue, Eigen::Vector3i(c, c, c), 1.0f);
		_fluidGrid->setBuoyancy(1);
		_fluidGrid->setVorticityCoeff(0.2f);
		//_fluidGrid->setHeatDiffusion(0.1f);

		// Setup the fluid rendering
		Shader smokeVert{ ShaderType::VertexShader,   0, smoke_vert_shader };
		Shader smokeGeom{ ShaderType::GeometryShader, 0, smoke_geom_shader };
		Shader smokeFrag{ ShaderType::FragmentShader, 0, smoke_frag_shader };
		PipelineStateDescription smokePSDesc;
		smokePSDesc.VertexShader   = &smokeVert;
		smokePSDesc.GeometryShader = &smokeGeom;
		smokePSDesc.FragmentShader = &smokeFrag;
		_smokePipelineState = std::make_unique<PipelineState>(smokePSDesc);

		initializeRenderVolume();
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
		// Update the fluid grid
		_fluidSolver->update(*_fluidGrid, 0.016f);

		// Copy the data from the fluid grid to OpenGL
		CUDA_MEMCPY3D cpy_desc;
		cpy_desc.srcXInBytes = 0;
		cpy_desc.srcY = 0;
		cpy_desc.srcZ = 0;
		cpy_desc.srcLOD = 0;
		cpy_desc.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
		cpy_desc.srcDevice = _fluidGrid->densities(0)->devicePtr();
		cpy_desc.srcPitch = 128 * sizeof(float);
		cpy_desc.srcHeight = 128;

		cpy_desc.dstXInBytes = 0;
		cpy_desc.dstY = 0;
		cpy_desc.dstZ = 0;
		cpy_desc.dstLOD = 0;
		cpy_desc.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
		cpy_desc.dstArray = _volumeArray;

		cpy_desc.WidthInBytes = 128 * sizeof(float);
		cpy_desc.Height = 128;
		cpy_desc.Depth = 128;

		VCL_CU_SAFE_CALL(cuMemcpy3DAsync(&cpy_desc, 0));
		_cudaContext->sync();

		// Draw the content
		glClearColor(0, 0, 0, 1);
		glClearDepth(1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		Eigen::Matrix4f vp = _camera->projection() * _camera->view();
		Eigen::Matrix4f m = _cameraController->currObjectTransformation();
		Eigen::AlignedBox3f bb{ Eigen::Vector3f{-10.0f, -10.0f, -10.0f }, Eigen::Vector3f{ 10.0f, 10.0f, 10.0f} };

		// Render the debug bounding grid
		renderBoundingBox(bb, _gridResolution, _boxPipelineState.get(), m, vp);

		// Render the smoke
		renderSmokeBox(bb, _smokePipelineState.get(), m, vp);
	}

private:
	void initializeCuda()
	{
		Vcl::Compute::Cuda::Platform::initialise();
		Vcl::Compute::Cuda::Platform* platform = Vcl::Compute::Cuda::Platform::instance();
		const Vcl::Compute::Cuda::Device& device = platform->device(0);
		_cudaContext = Vcl::make_owner<Vcl::Compute::Cuda::Context>(device, Vcl::Compute::Cuda::ApiBinding::OpenGL);
		_cudaContext->bind();
	}

	void initializeRenderVolume()
	{
		using namespace Vcl::Graphics::Runtime;

		// Create the volume texture sampler used to render the smoke
		SamplerDescription sampler_desc;
		sampler_desc.Filter = FilterType::MinMagLinearMipPoint;

		_volumeSampler = Vcl::make_owner<OpenGL::Sampler>(sampler_desc);

		// Create the volume texture used to render the smoke
		Texture3DDescription desc;
		desc.Width = 128;
		desc.Height = 128;
		desc.Depth = 128;
		desc.Format = Vcl::Graphics::SurfaceFormat::R32_FLOAT;
		desc.MipLevels = 1;

		_volumeTexture = Vcl::make_owner<OpenGL::Texture3D>(desc);

		// Register the volume texture with CUDA.
		// See https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
		cuGraphicsGLRegisterImage(&_volumeTextureResource, _volumeTexture->id(), GL_TEXTURE_3D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
		cuGraphicsMapResources(1, &_volumeTextureResource, 0);
		cuGraphicsSubResourceGetMappedArray(&_volumeArray, _volumeTextureResource, 0, 0);
		cuGraphicsUnmapResources(1, &_volumeTextureResource, 0);
	}

	void renderBoundingBox
	(
		const Eigen::AlignedBox3f& bb,
		unsigned int resolution,
		Vcl::Graphics::Runtime::OpenGL::PipelineState* ps,
		const Eigen::Matrix4f& M,
		const Eigen::Matrix4f& VP
	)
	{
		// Configure the layout
		ps->bind();

		// View on the scene
		ps->program().setUniform("ModelMatrix", M);
		ps->program().setUniform("ViewProjectionMatrix", VP);

		// Compute the grid paramters
		float maxSize = bb.diagonal().maxCoeff();
		Eigen::Vector3f origin = bb.center() - 0.5f * maxSize * Eigen::Vector3f::Ones().eval();

		ps->program().setUniform("Origin", origin);
		ps->program().setUniform("StepSize", maxSize / (float)resolution);
		ps->program().setUniform("Resolution", (float)resolution);
		
		// Render the grid in grey
		ps->program().setUniform("Colours[0]", Eigen::Vector3f::Constant(0.7f).eval());
		ps->program().setUniform("Colours[1]", Eigen::Vector3f::Constant(0.7f).eval());
		ps->program().setUniform("Colours[2]", Eigen::Vector3f::Constant(0.7f).eval());
		
		// Render the grid
		// 3 Line-loops with 4 points, N+1 replications of the loops (N tiles)
		glDrawArraysInstanced(GL_LINES_ADJACENCY, 0, 12, resolution + 1);
	}

	void renderSmokeBox
	(
		const Eigen::AlignedBox3f& bb,
		Vcl::Graphics::Runtime::OpenGL::PipelineState* ps,
		const Eigen::Matrix4f& M,
		const Eigen::Matrix4f& VP
	)
	{
		// Configure the layout
		ps->bind();

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);

		// View on the scene
		ps->program().setUniform("ModelMatrix", M);
		ps->program().setUniform("ViewProjectionMatrix", VP);

		// Compute the cube paramters
		float maxSize = bb.diagonal().maxCoeff();
		Eigen::Vector3f origin = bb.center() - 0.5f * maxSize * Eigen::Vector3f::Ones().eval();

		ps->program().setUniform("Origin", origin);
		
		// Position of the camera
		Eigen::Vector4f cam_pos;
		cam_pos.segment<3>(0) = _camera->position();
		cam_pos.w() = 1;
		Eigen::Vector3f cam_pos_ms = (M.inverse() * cam_pos).segment<3>(0);
		ps->program().setUniform("ViewPositionMS", cam_pos_ms);

		// Assign the volume to the shader
		ps->program().setTexture(ps->program().uniform("Density"), _volumeTexture.get(), _volumeSampler.get());

		// Render the cube
		glDrawArraysInstanced(GL_LINES_ADJACENCY, 0, 4, 6);
	}

private:
	std::unique_ptr<Vcl::Graphics::TrackballCameraController> _cameraController;

private:
	std::unique_ptr<Vcl::Graphics::Camera> _camera;
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _boxPipelineState;
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _smokePipelineState;

private:
	unsigned int _gridResolution{ 1 };

private:
	Vcl::owner_ptr<Vcl::Compute::Cuda::Context> _cudaContext;
	std::unique_ptr<Vcl::Physics::Fluid::Cuda::EulerFluidSimulation> _fluidSolver;
	std::unique_ptr<Vcl::Physics::Fluid::Cuda::CenterGrid> _fluidGrid;

	CUgraphicsResource _volumeTextureResource;
	CUarray _volumeArray;

	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::Sampler> _volumeSampler;
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::Texture3D> _volumeTexture;
};

int main(int /* argc */, char ** /* argv */)
{
	try
	{
		nanogui::init();

		{
			nanogui::ref<RealTimeWaveletTurbulence> app = new RealTimeWaveletTurbulence();
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
