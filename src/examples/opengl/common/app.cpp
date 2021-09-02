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
#include "app.h"

// C++ standard library
#include <iostream>
#include <exception>
#include <stdexcept>

// VCL
#include <vcl/graphics/opengl/context.h>
#include <vcl/graphics/runtime/opengl/graphicsengine.h>

const int Application::NumberOfFrames = 3;

static void printGlfwError(int error, const char* description)
{
	fprintf(stdout, "Glfw Error %d: %s\n", error, description);
}

static void resizeGlfwWindow(GLFWwindow* window, int width, int height)
{
	auto app = static_cast<Application*>(glfwGetWindowUserPointer(window));
	app->resize(width, height);
}

Application::Application(const char* title)
{
	glfwSetErrorCallback(printGlfwError);
	if (!glfwInit())
		throw std::runtime_error("Could not initialize GLFW");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	_windowHandle = glfwCreateWindow(1280, 720, title, nullptr, nullptr);
	if (_windowHandle == nullptr)
		throw std::runtime_error("Could not initialize GLFW window");

	glfwSetWindowUserPointer(_windowHandle, this);
	glfwSetWindowSizeCallback(_windowHandle, resizeGlfwWindow);
	//glfwSetMouseButtonCallback(_windowHandle, onMouseButton);
	//glfwSetCursorPosCallback(_windowHandle, onMouseMove);

	glfwMakeContextCurrent(_windowHandle);
	glfwSwapInterval(1); // Enable V-Sync

	// Setup OpenGL environment
	Vcl::Graphics::OpenGL::Context::initExtensions();
	Vcl::Graphics::OpenGL::Context::setupDebugMessaging();

	_engine = std::make_unique<Vcl::Graphics::Runtime::OpenGL::GraphicsEngine>();
}

Application::~Application()
{
	if (_windowHandle)
		glfwDestroyWindow(_windowHandle);
	glfwTerminate();
}

int Application::run()
{
	while (!glfwWindowShouldClose(windowHandle()))
	{
		glfwPollEvents();
		updateFrame();

		glfwMakeContextCurrent(windowHandle());

		_engine->beginFrame();
		renderFrame(*_engine);
		_engine->endFrame();

		glfwMakeContextCurrent(windowHandle());
		glfwSwapBuffers(windowHandle());
	}

	return 0;
}
