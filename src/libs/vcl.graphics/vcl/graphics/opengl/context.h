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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

// VCL

#if defined VCL_OPENGL_SUPPORT

// EGL
#	if defined VCL_EGL_SUPPORT
#		include <EGL/egl.h>
#		undef Success
#	endif

namespace Vcl { namespace Graphics { namespace OpenGL
{
	enum class ContextType
	{
		Core,
		Compatibility,
		Embedded
	};

	struct ContextDesc
	{
		int MajorVersion{ 3 };
		int MinorVersion{ 0 };
		ContextType Type{ ContextType::Compatibility };
		bool Debug{ false };
	};

	class Context final
	{
	public:
		//! Access the context type
		//! \returns The context type ('Core', 'Compatibility', 'Invalid')
		static const char* profileType();
		
		//! Initialize the OpenGL extension function pointers
		static void initExtensions();

		//! Enable the OpenGL debug message extension
		static void setupDebugMessaging();

	public:
		Context(const ContextDesc& desc = {});
#	if defined VCL_EGL_SUPPORT
		Context(EGLDisplay display, EGLSurface surface, const ContextDesc& desc = {});
#	endif
		~Context();

		//! Access the context descriptor
		//! \returns The context descriptor
		const ContextDesc& desc() const { return _desc; }

		//! Make the context the thread's current context
		//! \returns True, if the operation was successful
		bool makeCurrent();

	private:
#	if defined VCL_EGL_SUPPORT
		//! Associated EGL display
		EGLDisplay _display;

		//! EGL surface
		EGLSurface _surface;

		//! EGL context
		EGLContext _context;
#	elif defined VCL_ABI_WINAPI
		//! Windows window handle
		void* _window_handle{ nullptr };

		//! Windows display context
		void* _display_ctx{ nullptr };

		//! Windows render context
		void* _render_ctx{ nullptr };
#	endif

		//! Context description
		ContextDesc _desc;

		//! Allocated display
		bool _allocated_display{ false };
		
		//! Allocated surface
		bool _allocated_surface{ false };
	};
}}}
#endif // defined VCL_OPENGL_SUPPORT
