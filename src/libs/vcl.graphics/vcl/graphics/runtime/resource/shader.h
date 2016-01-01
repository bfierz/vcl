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

namespace Vcl { namespace Graphics { namespace Runtime
{
	enum class ShaderType
	{
		VertexShader,     //!< Shader executing programs on single vertices
		ControlShader,    //!< Shader executing programs on entire tesellation patches
		EvaluationShader, //!< Shader executing programs on each vertex generated from tessellation
		GeometryShader,   //!< Shader executing programs on entire primitives (lines, triangles)
		FragmentShader,   //!< Shader executing programs on single fragments
		ComputeShader     //!< Shader executing generic programs
	};

	class Shader
	{
	protected:
		Shader(ShaderType type, int tag);

	public:
		Shader(const Shader& rhs) = default;
		virtual ~Shader() = default;

	private:
		//! Tag identifying the owner
		int _tag;

		//! Type of this shader
		ShaderType _type;
	};
}}}
