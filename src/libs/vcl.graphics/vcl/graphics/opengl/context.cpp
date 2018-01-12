/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#include <vcl/graphics/opengl/context.h>

// C++ standard library
#include <iostream>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/opengl/gl.h>

#ifdef VCL_OPENGL_SUPPORT

// OpenGL
#include <GL/glew.h>

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
		VCL_UNREFERENCED_PARAMETER(length);
		VCL_UNREFERENCED_PARAMETER(user_param);

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

namespace Vcl { namespace Graphics { namespace OpenGL
{
	const char* Context::profileType()
	{
		GLint profile = GL::getInteger(GL_CONTEXT_PROFILE_MASK);

		if (profile == GL_CONTEXT_CORE_PROFILE_BIT)
			return "Core";
		else if (profile == GL_CONTEXT_COMPATIBILITY_PROFILE_BIT)
			return "Compatibility";
		else
			return "Invalid";
	}

	Context::Context(EGLDisplay display, EGLSurface surface, const ContextDesc& desc)
	: _desc(desc)
	{
		// 1. Create a surface if non is supplied
		if (!display)
		{
			_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
			_allocated_display = true;

			EGLint major, minor;
			eglInitialize(_display, &major, &minor);

			display = _display;
		}
		
		// 2. Select an appropriate configuration
		// 3. Create a surface
		EGLConfig egl_config;
		if (!surface)
		{		
			const EGLint surface_config_attribs[] =
			{
				EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
				EGL_BLUE_SIZE, 8,
				EGL_GREEN_SIZE, 8,
				EGL_RED_SIZE, 8,
				EGL_DEPTH_SIZE, 24,
				EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
				EGL_NONE
			};
	
			const int pbuffer_width = 32;
			const int pbuffer_height = 32;
	
			const EGLint pbuffer_attribs[] =
			{
				EGL_WIDTH, pbuffer_width,
				EGL_HEIGHT, pbuffer_height,
				EGL_NONE,
			};

			EGLint num_configs;	  
			eglChooseConfig(_display, surface_config_attribs, &egl_config, 1, &num_configs);
	  
			_surface = eglCreatePbufferSurface(_display, egl_config, pbuffer_attribs);
			_allocated_surface = true;	  
		}

		// 4. Bind the API
		eglBindAPI(EGL_OPENGL_API);
	  
		// 5. Create a eglContext and make it current
		const EGLint context_attribute[] =
		{
			EGL_CONTEXT_MAJOR_VERSION, desc.MajorVersion,
			EGL_CONTEXT_MINOR_VERSION, desc.MinorVersion,
			EGL_CONTEXT_OPENGL_PROFILE_MASK, desc.Type == ContextType::Core ? EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT : EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
			EGL_NONE
		};
		_context = eglCreateContext(_display, egl_config, EGL_NO_CONTEXT, context_attribute);
		makeCurrent();
	  		
		initExtensions();

		if (desc.Debug)
			setupDebugMessaging();
	}

	Context::~Context()
	{
		eglDestroyContext(_display, _context);
		if (_allocated_surface)
		{
			eglDestroySurface(_display, _surface);
		}
		if (_allocated_display)
		{
			eglTerminate(_display);
		}
	}
	
	bool Context::makeCurrent()
	{
		return eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, _context);
	}

	void Context::initExtensions()
	{
		// Initialize glew
		glewExperimental = GL_TRUE;
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			std::cout << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
			std::terminate();
		}

		std::cout << "Status: Using OpenGL:   " << glGetString(GL_VERSION) << std::endl;
		std::cout << "Status:       Vendor:   " << glGetString(GL_VENDOR) << std::endl;
		std::cout << "Status:       Renderer: " << glGetString(GL_RENDERER) << std::endl;
		std::cout << "Status:       Profile:  " << profileType() << std::endl;
		std::cout << "Status:       Shading:  " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
		std::cout << "Status: Using GLEW:     " << glewGetString(GLEW_VERSION) << std::endl;
	}

	void Context::setupDebugMessaging()
	{
		// Enable the synchronous debug output
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

		// Disable debug severity: notification
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);

		// Disable specific messages
		GLuint perf_messages_ids[] =
		{
			131154, // Pixel-path performance warning: Pixel transfer is synchronized with 3D rendering
		//	131218, // NVIDIA: "shader will be recompiled due to GL state mismatches"
		};
		glDebugMessageControl
		(
			GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE,
			sizeof(perf_messages_ids) / sizeof(GLuint), perf_messages_ids, GL_FALSE
		);

		// Register debug callback
		glDebugMessageCallback(OpenGLDebugMessageCallback, nullptr);
	}
}}}
#endif // VCL_OPENGL_SUPPORT
