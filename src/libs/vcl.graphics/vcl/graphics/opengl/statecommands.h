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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

// C++ standard library
#include <vector>

// VCL

namespace Vcl { namespace Graphics { namespace OpenGL
{
	enum class CommandType
	{
		Enable,
		Disable,
		Enablei,
		Disablei,
		LogicOp,
		BlendFunc,
		BlendEquation,
		BlendFunci,
		BlendEquationi,
		BlendColor,
		ColorMask,
		ColorMaskIndexed,
		DepthMask,
		DepthFunc,
		StencilOpSeparate,
		StencilMaskSeparate,
		CullFace,
		FrontFace,
		PolygonMode,
		PolygonOffsetClamp,
		PolygonOffset
	};

	class StateCommands
	{
	public:
		template<typename... Args>
		void emplace(CommandType type, Args&&... args)
		{
			addTokens(type, { toToken(args)... });
		}

	public:
		void bind();
		void unbind();

	private:
		uint32_t toToken(int arg);
		uint32_t toToken(float arg);
		uint32_t toToken(GLenum arg);
		float toFloat(uint32_t tok);
		void addTokens(CommandType type, std::initializer_list<uint32_t> params);

	private:
		std::vector<uint32_t> _commands;
	};
}}}
