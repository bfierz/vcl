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
#include <vcl/config/eigen.h>
#include <vcl/config/opengl.h>

// C++ standard libary
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/resource/buffer.h>
#include <vcl/graphics/runtime/state/inputlayout.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	template<typename VertexDataType>
	struct VertexDataTypeTrait
	{
		static const GLenum Type;
		static const GLint Size;
		static const GLint NrComponents;
		static const bool IsIntegral;
	};

	struct VertexLayoutElement
	{
		GLint NrComponents;
		GLenum ElementType;
		GLsizei ElementSize;
		bool IsIntegral;
	};

	class InputLayout
	{
	public:
		explicit InputLayout(const Runtime::InputLayoutDescription& desc);
		~InputLayout();

	public:
		void bind();

	private:
		void setup(const Runtime::InputLayoutDescription& desc);

	private:
		//! Description
		InputLayoutDescription _desc;

		//! Vertex array object that caches the input layout
		GLuint _vaoID{ 0 };
	};

	// Extern template specialization
#ifndef VCL_GRAPHICS_HELIOS_OPENGL_INPUTLAYOUT_INST
	extern template	struct VertexDataTypeTrait<float>;
	extern template	struct VertexDataTypeTrait<Eigen::Vector2f>;
	extern template	struct VertexDataTypeTrait<Eigen::Vector3f>;
	extern template	struct VertexDataTypeTrait<Eigen::Vector4f>;
	extern template	struct VertexDataTypeTrait<int>;
	extern template	struct VertexDataTypeTrait<Eigen::Vector2i>;
	extern template	struct VertexDataTypeTrait<Eigen::Vector3i>;
	extern template	struct VertexDataTypeTrait<Eigen::Vector4i>;
#endif // VCL_GRAPHICS_HELIOS_OPENGL_INPUTLAYOUT_INST
}}}}
