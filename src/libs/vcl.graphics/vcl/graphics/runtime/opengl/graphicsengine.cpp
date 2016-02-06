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

#include <vcl/graphics/runtime/opengl/graphicsengine.h>

// C++ standard library
#include <iostream>

// VCL
#include <vcl/graphics/opengl/gl.h>

namespace
{
	void VCL_CALLBACK OpenGLDebugMessageCallback
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

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Fence::Fence(GLsync sync)
	: _sync(sync)
	{

	}

	Fence::Fence(Fence&& rhs)
	{
		std::swap(_sync, rhs._sync);
	}

	Fence::~Fence()
	{
		glDeleteSync(_sync);
		_sync = nullptr;
	}

	Fence& Fence::operator=(Fence&& rhs)
	{
		std::swap(_sync, rhs._sync);
		return *this;
	}

	Frame::Frame()
	{
		BufferDescription cbuffer_desc;

		// Allocate a junk of 512 KB for constant buffers per frame
		cbuffer_desc.SizeInBytes = 1 << 19;
		cbuffer_desc.CPUAccess = CPUAccess::Write;
		cbuffer_desc.Usage = Usage::Dynamic;

		_constantBuffer = make_owner<Buffer>(cbuffer_desc, true, true);
	}

	void Frame::mapConstantBuffer()
	{
		_mappedConstantBuffer = _constantBuffer->map(0, _constantBuffer->sizeInBytes(), CPUAccess::Write, MapOptions::Persistent | MapOptions::CoherentWrite | MapOptions::Unsynchronized);
	}

	void Frame::unmapConstantBuffer()
	{
		_constantBuffer->unmap();
	}

	GraphicsEngine::GraphicsEngine()
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

		// Read the device dependent constant
		_cbufferAlignment = GL::getInteger(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT);

		// Create three frames:
		// 1. where the CPU writes to
		// 2. which the GPU driver is processing
		// 3. which the GPU is processing
		_frames.resize(_numConcurrentFrames);
	}

	void GraphicsEngine::beginFrame()
	{
		// Increment the frame counter to indicate the start of the next frame
		incrFrameCounter();

		// Fetch the frame we want to use for the current rendering frame
		auto& curr_frame = _frames[currentFrame() % _numConcurrentFrames];

		// Wait until the requested frame is done with processing
		if (curr_frame.fence()->isValid())
		{
			auto wait_result = glClientWaitSync(*curr_frame.fence(), GL_NONE, 1000000000);
			if (wait_result != GL_CONDITION_SATISFIED || wait_result != GL_ALREADY_SIGNALED)
			{
				if (wait_result == GL_TIMEOUT_EXPIRED)
					throw std::runtime_error("Waiting for sync object expired.");

				if (wait_result == GL_WAIT_FAILED)
					throw std::runtime_error("Waiting for sync object failed.");
			}
		}

		// Map the per frame constant buffer
		curr_frame.mapConstantBuffer();

		// Reset the begin pointer for constant buffer chunk requests
		_cbufferOffset = 0;
	}

	void GraphicsEngine::endFrame()
	{
		// Fetch the frame we are using for the current rendering frame
		auto& curr_frame = _frames[currentFrame() % _numConcurrentFrames];

		// Unmap the constant buffer
		curr_frame.unmapConstantBuffer();

		// Queue a fence sync object to mark the end of the frame
		curr_frame.setFence(glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, GL_NONE));
	}

	BufferView GraphicsEngine::requestPerFrameConstantBuffer(size_t size)
	{
		// Fetch the frame we are using for the current rendering frame
		auto& curr_frame = _frames[currentFrame() % _numConcurrentFrames];

		BufferView view{ curr_frame.constantBuffer(), _cbufferOffset, size, curr_frame.mappedConstantBuffer() };

		// Calculate the next offset		
		size_t aligned_size =  ((size + (_cbufferAlignment - 1)) / _cbufferAlignment) * _cbufferAlignment;
		_cbufferOffset += aligned_size;

		return view;
	}
}}}}
