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
#define VCL_GRAPHICS_OPENGL_RENDERTYPETRAITS_INST
#include <vcl/graphics/opengl/type_traits.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace OpenGL
{
	const GLenum AnyRenderType::componentType() const { return _gate().componentType(); }
	const GLint AnyRenderType::size() const { return _gate().size(); }
	const GLint AnyRenderType::componentSize() const { return _gate().componentSize(); }
	const GLint AnyRenderType::nrComponents() const { return _gate().nrComponents(); }
	const bool AnyRenderType::isIntegral() const { return _gate().isIntegral(); }

	template<>
	struct RenderTypeTrait<void>
	{
		typedef void Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<void>::ComponentType = GL_NONE;
	const GLint  RenderTypeTrait<void>::ComponentSize = 1;
	const GLint  RenderTypeTrait<void>::NrComponents = 1;
	const GLint  RenderTypeTrait<void>::Size = 1;
	const bool   RenderTypeTrait<void>::IsIntegral = false;

	template<>
	struct RenderTypeTrait<float>
	{
		typedef float Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<float>::ComponentType = GL_FLOAT;
	const GLint  RenderTypeTrait<float>::ComponentSize = sizeof(float);
	const GLint  RenderTypeTrait<float>::NrComponents = 1;
	const GLint  RenderTypeTrait<float>::Size = sizeof(float);
	const bool   RenderTypeTrait<float>::IsIntegral = false;

	template<>
	struct RenderTypeTrait<Eigen::Vector2f>
	{
		typedef Eigen::Vector2f Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<Eigen::Vector2f>::ComponentType = GL_FLOAT;
	const GLint  RenderTypeTrait<Eigen::Vector2f>::ComponentSize = sizeof(Eigen::Vector2f::Scalar);
	const GLint  RenderTypeTrait<Eigen::Vector2f>::NrComponents = 2;
	const GLint  RenderTypeTrait<Eigen::Vector2f>::Size = sizeof(Eigen::Vector2f);
	const bool   RenderTypeTrait<Eigen::Vector2f>::IsIntegral = false;

	template<>
	struct RenderTypeTrait<Eigen::Vector3f>
	{
		typedef Eigen::Vector3f Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<Eigen::Vector3f>::ComponentType = GL_FLOAT;
	const GLint  RenderTypeTrait<Eigen::Vector3f>::ComponentSize = sizeof(Eigen::Vector3f::Scalar);
	const GLint  RenderTypeTrait<Eigen::Vector3f>::NrComponents = 3;
	const GLint  RenderTypeTrait<Eigen::Vector3f>::Size = sizeof(Eigen::Vector3f);
	const bool   RenderTypeTrait<Eigen::Vector3f>::IsIntegral = false;

	template<>
	struct RenderTypeTrait<Eigen::Vector4f>
	{
		typedef Eigen::Vector4f Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<Eigen::Vector4f>::ComponentType = GL_FLOAT;
	const GLint  RenderTypeTrait<Eigen::Vector4f>::ComponentSize = sizeof(Eigen::Vector4f::Scalar);
	const GLint  RenderTypeTrait<Eigen::Vector4f>::NrComponents = 4;
	const GLint  RenderTypeTrait<Eigen::Vector4f>::Size = sizeof(Eigen::Vector4f);
	const bool   RenderTypeTrait<Eigen::Vector4f>::IsIntegral = false;

	template<>
	struct RenderTypeTrait<int>
	{
		typedef int Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<int>::ComponentType = GL_INT;
	const GLint  RenderTypeTrait<int>::ComponentSize = sizeof(int);
	const GLint  RenderTypeTrait<int>::NrComponents = 1;
	const GLint  RenderTypeTrait<int>::Size = sizeof(int);
	const bool   RenderTypeTrait<int>::IsIntegral = true;

	template<>
	struct RenderTypeTrait<Eigen::Vector2i>
	{
		typedef Eigen::Vector2i Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<Eigen::Vector2i>::ComponentType = GL_INT;
	const GLint  RenderTypeTrait<Eigen::Vector2i>::ComponentSize = sizeof(Eigen::Vector2i::Scalar);
	const GLint  RenderTypeTrait<Eigen::Vector2i>::NrComponents = 2;
	const GLint  RenderTypeTrait<Eigen::Vector2i>::Size = sizeof(Eigen::Vector2i);
	const bool   RenderTypeTrait<Eigen::Vector2i>::IsIntegral = true;

	template<>
	struct RenderTypeTrait<Eigen::Vector3i>
	{
		typedef Eigen::Vector3i Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<Eigen::Vector3i>::ComponentType = GL_INT;
	const GLint  RenderTypeTrait<Eigen::Vector3i>::ComponentSize = sizeof(Eigen::Vector3i::Scalar);
	const GLint  RenderTypeTrait<Eigen::Vector3i>::NrComponents = 3;
	const GLint  RenderTypeTrait<Eigen::Vector3i>::Size = sizeof(Eigen::Vector3i);
	const bool   RenderTypeTrait<Eigen::Vector3i>::IsIntegral = true;

	template<>
	struct RenderTypeTrait<Eigen::Vector4i>
	{
		typedef Eigen::Vector4i Type;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<Eigen::Vector4i>::ComponentType = GL_INT;
	const GLint  RenderTypeTrait<Eigen::Vector4i>::ComponentSize = sizeof(Eigen::Vector4i::Scalar);
	const GLint  RenderTypeTrait<Eigen::Vector4i>::NrComponents = 4;
	const GLint  RenderTypeTrait<Eigen::Vector4i>::Size = sizeof(Eigen::Vector4i);
	const bool   RenderTypeTrait<Eigen::Vector4i>::IsIntegral = true;
}}}
#endif // VCL_OPENGL_SUPPORT
