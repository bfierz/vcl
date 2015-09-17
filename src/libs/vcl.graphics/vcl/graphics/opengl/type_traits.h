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
#include <vcl/config/eigen.h>

#ifdef VCL_OPENGL_SUPPORT

VCL_BEGIN_EXTERNAL_HEADERS
#include <GL/glew.h>
VCL_END_EXTERNAL_HEADERS

// VCL
#include <vcl/graphics/surfaceformat.h>
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace OpenGL
{
	template<typename T>
	struct RenderTypeTrait
	{
		using Type = T;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};

	class AnyRenderType
	{
	public:
		typedef const AnyRenderType& (*DynamicRenderTypeGate)();

	protected:
		AnyRenderType(DynamicRenderTypeGate gate) : _gate(gate) { Ensure(_gate != nullptr, "gate not NULL"); }

	public:
		virtual const GLenum componentType() const;
		virtual const GLint size() const;
		virtual const GLint componentSize() const;
		virtual const GLint nrComponents() const;
		virtual const bool isIntegral() const;

		//virtual const bool isPixelType() const { return _gate().isPixelType(); }
		//virtual const GLenum internalFormat() const { Require(isPixelType(), "type is color type"); return _gate().internalFormat(); }
		//virtual const GLenum internalBaseFormat() const { Require(isPixelType(), "type is color type"); return _gate().internalBaseFormat(); }
		//virtual const GLenum format() const { Require(isPixelType(), "type is color type"); return _gate().format(); }
		//virtual const bool hasAlphaChannel() const { Require(isPixelType(), "type is color type"); return _gate().hasAlphaChannel(); }

	private:
		DynamicRenderTypeGate _gate;
	};
	
	template <typename T> class RenderType;

	template <typename T>
	struct DynamicRenderType
	{
		static const AnyRenderType& warp() { static const RenderType<T> obj; return obj; }
	};

	template <typename T>
	class RenderType : public AnyRenderType
	{
	public:
		typedef T Type;
		typedef RenderTypeTrait<T> TypeTrait;		

	public:
		RenderType() : AnyRenderType(&DynamicRenderType<T>::warp) {}

		const GLenum componentType() const override { return TypeTrait::ComponentType; }
		const GLint size() const override { return TypeTrait::Size; }
		const GLint componentSize() const override { return TypeTrait::ComponentSize; }
		const GLint nrComponents() const override { return TypeTrait::NrComponents; }
		const bool isIntegral() const override { return TypeTrait::IsIntegral; }
		
		//const bool isPixelType() const override {return TypeTrait::IsPixelType; }
		//const GLenum internalFormat() const override {Require(isPixelType(), "type is pixel/color type"); return TypeTrait::IsPixelType?TypeTrait::InternalFormat:GL_NONE; }
		//const GLenum internalBaseFormat() const override {Require(isPixelType(), "type is pixel/color type"); return TypeTrait::IsPixelType?TypeTrait::InternalBaseFormat:GL_NONE; }
		//const GLenum format() const override {Require(isPixelType(), "type is pixel/color type"); return TypeTrait::IsPixelType?TypeTrait::Format:GL_NONE; }
		//const bool hasAlphaChannel() const override {Require(isPixelType(), "type is pixel/color type"); return TypeTrait::IsPixelType?TypeTrait::HasAlphaChannel:false; }
	};

#ifndef VCL_GRAPHICS_OPENGL_RENDERTYPETRAITS_INST
	extern template struct RenderTypeTrait<void>;

	extern template struct RenderTypeTrait<float>;
	extern template struct RenderTypeTrait<Eigen::Vector2f>;
	extern template struct RenderTypeTrait<Eigen::Vector3f>;
	extern template struct RenderTypeTrait<Eigen::Vector4f>;
	
	extern template struct RenderTypeTrait<int>;
	extern template struct RenderTypeTrait<Eigen::Vector2i>;
	extern template struct RenderTypeTrait<Eigen::Vector3i>;
	extern template struct RenderTypeTrait<Eigen::Vector4i>;
#endif /* VCL_GRAPHICS_OPENGL_RENDERTYPETRAITS_INST */
}}}
#endif // VCL_OPENGL_SUPPORT
