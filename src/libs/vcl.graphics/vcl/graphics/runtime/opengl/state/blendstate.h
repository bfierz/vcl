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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

// C++ standard library
#include <array>

// VCL
#include <vcl/graphics/opengl/statecommands.h>
#include <vcl/graphics/runtime/state/blendstate.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	/*!
	 *	\brief OpenGL abstraction of the blending related pipeline states
	 */
	class BlendState
	{
	public:
		BlendState(const BlendDescription& desc);

	public:
		const BlendDescription& desc() const { return _desc; }

	public:
		/*!
		 * \brief Bind the blend configurations
		 */
		void bind();

		/*!
		 * \brief Append the state changes to the state command buffer
		 */
		void record(Graphics::OpenGL::StateCommands& states);

	public:
		bool isValid() const;

	private: // Debug
		bool check() const;

	public:
		static bool isIndependentBlendingSupported();
		static bool areAdvancedBlendOperationsSupported();

	public:
		static GLenum toGLenum(BlendOperation op);
		static GLenum toGLenum(LogicOperation op);
		static GLenum toGLenum(Blend factor);

	private:
		//! Description of the blend state
		BlendDescription _desc;
	};
}}}}
#endif // VCL_OPENGL_SUPPORT
