/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#include "meshview.h"

// Qt
#include <QtQuick/QQuickWindow>

// VCL
#include <vcl/graphics/opengl/gl.h>
#include <vcl/graphics/runtime/opengl/resource/shader.h>

#include "scene.h"

namespace
{
	void CALLBACK OpenGLDebugMessageCallback
	(
		GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const GLchar* message,
		const void* user_param
	)
	{
		// Suppress some useless warnings
		switch (id)
		{
		case 131218: // NVIDIA: "shader will be recompiled due to GL state mismatches"
			return;
		default:
			break;
		}

		std::cout << "Source: ";
		switch (source)
		{
		case GL_DEBUG_SOURCE_API:
			std::cout << "API";
			break;
		case GL_DEBUG_SOURCE_SHADER_COMPILER:
			std::cout << "Shader Compiler";
			break;
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
			std::cout << "Window System";
			break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:
			std::cout << "Third Party";
			break;
		case GL_DEBUG_SOURCE_APPLICATION:
			std::cout << "Application";
			break;
		case GL_DEBUG_SOURCE_OTHER:
			std::cout << "Other";
			break;
		}

		std::cout << ", Type: ";
		switch (type)
		{
		case GL_DEBUG_TYPE_ERROR:
			std::cout << "Error";
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			std::cout << "Deprecated Behavior";
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			std::cout << "Undefined Behavior";
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			std::cout << "Performance";
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			std::cout << "Portability";
			break;
		case GL_DEBUG_TYPE_OTHER:
			std::cout << "Other";
			break;
		case GL_DEBUG_TYPE_MARKER:
			std::cout << "Marker";
			break;
		case GL_DEBUG_TYPE_PUSH_GROUP:
			std::cout << "Push Group";
			break;
		case GL_DEBUG_TYPE_POP_GROUP:
			std::cout << "Pop Group";
			break;
		}

		std::cout << ", Severity: ";
		switch (severity)
		{
		case GL_DEBUG_SEVERITY_HIGH:
			std::cout << "High";
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			std::cout << "Medium";
			break;
		case GL_DEBUG_SEVERITY_LOW:
			std::cout << "Low";
			break;
		case GL_DEBUG_SEVERITY_NOTIFICATION:
			std::cout << "Notification";
			break;
		}

		std::cout << ", ID: " << id;
		std::cout << ", Message: " << message << std::endl;
	}
}

namespace
{
	Vcl::Graphics::Runtime::OpenGL::Shader createShader(Vcl::Graphics::Runtime::ShaderType type, QString path)
	{
		QFile shader_file{ path };
		shader_file.open(QIODevice::ReadOnly);
		Check(shader_file.isOpen(), "Shader file is open.");

		return{ type, 0, shader_file.readAll().data() };
	}
}

FboRenderer::FboRenderer()
{
	using Vcl::Graphics::Runtime::OpenGL::InputLayout;
	using Vcl::Graphics::Runtime::OpenGL::Shader;
	using Vcl::Graphics::Runtime::OpenGL::ShaderProgramDescription;
	using Vcl::Graphics::Runtime::OpenGL::ShaderProgram;
	using Vcl::Graphics::Runtime::InputLayoutDescription;
	using Vcl::Graphics::Runtime::ShaderType;
	using Vcl::Graphics::Runtime::VertexDataClassification;
	using Vcl::Graphics::SurfaceFormat;

	InputLayoutDescription opaqueTetraLayout =
	{
		{ "Indices", SurfaceFormat::R32G32B32A32_SINT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
		{ "Colour", SurfaceFormat::R32G32B32A32_FLOAT, 0, 1, 0, VertexDataClassification::VertexDataPerObject, 0 },
	};
	_opaqueTetraLayout = std::make_unique<InputLayout>(opaqueTetraLayout);

	Shader opaqueTetraVert = createShader(ShaderType::VertexShader, ":/shaders/tetramesh.vert");
	Shader opaqueTetraGeom = createShader(ShaderType::GeometryShader, ":/shaders/tetramesh.geom");

	Shader meshFrag = createShader(ShaderType::FragmentShader, ":/shaders/mesh.frag");

	ShaderProgramDescription opaqueTetraDesc;
	opaqueTetraDesc.InputLayout = opaqueTetraLayout;
	opaqueTetraDesc.VertexShader = &opaqueTetraVert;
	opaqueTetraDesc.GeometryShader = &opaqueTetraGeom;
	opaqueTetraDesc.FragmentShader = &meshFrag;

	_opaqueTetraMeshShader = std::make_unique<ShaderProgram>(opaqueTetraDesc);
}

void FboRenderer::render()
{
	glClearColor(0, 0, 0, 1);
	glClearDepth(1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (_owner)
	{
		auto scene = _owner->scene();

		////////////////////////////////////////////////////////////////////////
		// Prepare the environment
		////////////////////////////////////////////////////////////////////////

		Eigen::Matrix4f M = scene->modelMatrix();
		Eigen::Matrix4f V = scene->viewMatrix();
		Eigen::Matrix4f P = scene->projMatrix();
		
		_opaqueTetraMeshShader->setUniform(_opaqueTetraMeshShader->uniform("ModelMatrix"), M);
		_opaqueTetraMeshShader->setUniform(_opaqueTetraMeshShader->uniform("ViewMatrix"), V);
		_opaqueTetraMeshShader->setUniform(_opaqueTetraMeshShader->uniform("ProjectionMatrix"), P);

		// Configure the layout
		_opaqueTetraMeshShader->bind();
		_opaqueTetraLayout->bind();

		////////////////////////////////////////////////////////////////////////
		// Render the mesh
		////////////////////////////////////////////////////////////////////////

		// Set the vertex positions
		auto mesh = scene->volumeMesh();
		if (mesh)
		{
			_opaqueTetraMeshShader->setBuffer("VertexPositions", mesh->positions());

			// Bind the buffers
			glBindVertexBuffer(0, mesh->indices()->id(),       0, sizeof(Eigen::Vector4i));
			glBindVertexBuffer(1, mesh->volumeColours()->id(), 0, sizeof(Eigen::Vector4f));

			// Render the mesh
			glDrawArrays(GL_POINTS, 0, mesh->nrVolumes());
		}

		_owner->window()->resetOpenGLState();
	}

	update();
}

void FboRenderer::synchronize(QQuickFramebufferObject* item)
{
	auto* view = dynamic_cast<MeshView*>(item);
	if (view)
	{
		_owner = view;
	}

	if (_owner && _owner->scene())
	{
		_owner->scene()->update();
	}
}

QOpenGLFramebufferObject* FboRenderer::createFramebufferObject(const QSize &size)
{
	QOpenGLFramebufferObjectFormat format;
	format.setAttachment(QOpenGLFramebufferObject::Depth);
	format.setSamples(4);
	return new QOpenGLFramebufferObject(size, format);
}


MeshView::Renderer* MeshView::createRenderer() const
{
	using Vcl::Graphics::OpenGL::GL;

	// Initialize glew
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		std::cout << "Error: GLEW: " << glewGetErrorString(err) << std::endl;
	}

	std::cout << "Status: Using OpenGL:   " << glGetString(GL_VERSION) << std::endl;
	std::cout << "Status:       Vendor:   " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Status:       Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Status:       Profile:  " << GL::getProfileInfo() << std::endl;
	std::cout << "Status:       Shading:  " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
	std::cout << "Status: Using GLEW:     " << glewGetString(GLEW_VERSION) << std::endl;

	// Enable the synchronous debug output
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

	// Disable debug severity: notification
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);

	// Register debug callback
	glDebugMessageCallback(OpenGLDebugMessageCallback, nullptr);

	return new FboRenderer();
}

void MeshView::geometryChanged(const QRectF & newGeometry, const QRectF & oldGeometry)
{
	QQuickFramebufferObject::geometryChanged(newGeometry, oldGeometry);

	if (scene())
	{
		scene()->camera()->setViewport(newGeometry.width(), newGeometry.height());
	}
}
