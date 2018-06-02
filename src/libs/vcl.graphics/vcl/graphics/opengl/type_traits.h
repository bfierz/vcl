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
#include <vcl/config/opengl.h>

#ifdef VCL_OPENGL_SUPPORT

// VCL
#include <vcl/graphics/surfaceformat.h>
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace OpenGL
{
	template<typename T>
	struct RenderTypeTrait
	{
		using Type = T;
		static const GLenum InternalFormat;
		static const GLenum Format;
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
		AnyRenderType(DynamicRenderTypeGate gate) : _gate(gate) { VclEnsure(_gate != nullptr, "gate not NULL"); }

	public:
		virtual GLenum internalFormat() const;
		virtual GLenum format() const;
		virtual GLenum componentType() const;
		virtual GLint size() const;
		virtual GLint componentSize() const;
		virtual GLint nrComponents() const;
		virtual bool isIntegral() const;

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
		
		GLenum internalFormat() const override { return TypeTrait::InternalFormat; }
		GLenum format() const override { return TypeTrait::Format; }
		GLenum componentType() const override { return TypeTrait::ComponentType; }
		GLint size() const override { return TypeTrait::Size; }
		GLint componentSize() const override { return TypeTrait::ComponentSize; }
		GLint nrComponents() const override { return TypeTrait::NrComponents; }
		bool isIntegral() const override { return TypeTrait::IsIntegral; }
	};

#define VCL_GRAPHICS_RENDERTYPE_1(name, type) struct name    { using Type = type; Type x; };
#define VCL_GRAPHICS_RENDERTYPE_2(name, type) struct name##2 { using Type = type; Type x, y; };
#define VCL_GRAPHICS_RENDERTYPE_3(name, type) struct name##3 { using Type = type; Type x, y, z; };
#define VCL_GRAPHICS_RENDERTYPE_4(name, type) struct name##4 { using Type = type; Type x, y, z, w; }
#define VCL_GRAPHICS_RENDERTYPE(name, type)   \
		VCL_GRAPHICS_RENDERTYPE_1(name, type) \
		VCL_GRAPHICS_RENDERTYPE_2(name, type) \
		VCL_GRAPHICS_RENDERTYPE_3(name, type) \
		VCL_GRAPHICS_RENDERTYPE_4(name, type)

VCL_GRAPHICS_RENDERTYPE(Float, float);
VCL_GRAPHICS_RENDERTYPE(SignedInt, int);
VCL_GRAPHICS_RENDERTYPE(UnsignedInt, unsigned int);

VCL_GRAPHICS_RENDERTYPE(Half, short);
VCL_GRAPHICS_RENDERTYPE(SignedShort, short);
VCL_GRAPHICS_RENDERTYPE(UnsignedShort, unsigned short);
VCL_GRAPHICS_RENDERTYPE(NormalizedSignedShort, short);
VCL_GRAPHICS_RENDERTYPE(NormalizedUnsignedShort, unsigned short);

VCL_GRAPHICS_RENDERTYPE(SignedByte, char);
VCL_GRAPHICS_RENDERTYPE(UnsignedByte, unsigned char);
VCL_GRAPHICS_RENDERTYPE(NormalizedSignedByte, char);
VCL_GRAPHICS_RENDERTYPE(NormalizedUnsignedByte, unsigned char);

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

	extern template struct RenderTypeTrait<unsigned int>;
	extern template struct RenderTypeTrait<Eigen::Vector2ui>;
	extern template struct RenderTypeTrait<Eigen::Vector3ui>;
	extern template struct RenderTypeTrait<Eigen::Vector4ui>;

	extern template struct RenderTypeTrait<Float>;
	extern template struct RenderTypeTrait<Float2>;
	extern template struct RenderTypeTrait<Float3>;
	extern template struct RenderTypeTrait<Float4>;

	extern template struct RenderTypeTrait<SignedInt>;
	extern template struct RenderTypeTrait<SignedInt2>;
	extern template struct RenderTypeTrait<SignedInt3>;
	extern template struct RenderTypeTrait<SignedInt4>;

	extern template struct RenderTypeTrait<UnsignedInt>;
	extern template struct RenderTypeTrait<UnsignedInt2>; 
	extern template struct RenderTypeTrait<UnsignedInt3>; 
	extern template struct RenderTypeTrait<UnsignedInt4>; 

	extern template struct RenderTypeTrait<Half>;
	extern template struct RenderTypeTrait<Half2>;
	extern template struct RenderTypeTrait<Half3>;
	extern template struct RenderTypeTrait<Half4>;

	extern template struct RenderTypeTrait<short>;
	extern template struct RenderTypeTrait<SignedShort>;
	extern template struct RenderTypeTrait<SignedShort2>; 
	extern template struct RenderTypeTrait<SignedShort3>; 
	extern template struct RenderTypeTrait<SignedShort4>; 

	extern template struct RenderTypeTrait<unsigned short>;
	extern template struct RenderTypeTrait<UnsignedShort>;
	extern template struct RenderTypeTrait<UnsignedShort2>;
	extern template struct RenderTypeTrait<UnsignedShort3>;
	extern template struct RenderTypeTrait<UnsignedShort4>;

	extern template struct RenderTypeTrait<char>;
	extern template struct RenderTypeTrait<SignedByte>;
	extern template struct RenderTypeTrait<SignedByte2>;
	extern template struct RenderTypeTrait<SignedByte3>;
	extern template struct RenderTypeTrait<SignedByte4>;

	extern template struct RenderTypeTrait<unsigned char>;
	extern template struct RenderTypeTrait<UnsignedByte>;
	extern template struct RenderTypeTrait<UnsignedByte2>;
	extern template struct RenderTypeTrait<UnsignedByte3>;
	extern template struct RenderTypeTrait<UnsignedByte4>;

	extern template struct RenderTypeTrait<NormalizedSignedShort>;
	extern template struct RenderTypeTrait<NormalizedSignedShort2>;
	extern template struct RenderTypeTrait<NormalizedSignedShort3>;
	extern template struct RenderTypeTrait<NormalizedSignedShort4>;

	extern template struct RenderTypeTrait<NormalizedUnsignedShort>;
	extern template struct RenderTypeTrait<NormalizedUnsignedShort2>;
	extern template struct RenderTypeTrait<NormalizedUnsignedShort3>;
	extern template struct RenderTypeTrait<NormalizedUnsignedShort4>;

	extern template struct RenderTypeTrait<NormalizedSignedByte>;
	extern template struct RenderTypeTrait<NormalizedSignedByte2>;
	extern template struct RenderTypeTrait<NormalizedSignedByte3>;
	extern template struct RenderTypeTrait<NormalizedSignedByte4>;

	extern template struct RenderTypeTrait<NormalizedUnsignedByte>;
	extern template struct RenderTypeTrait<NormalizedUnsignedByte2>;
	extern template struct RenderTypeTrait<NormalizedUnsignedByte3>;
	extern template struct RenderTypeTrait<NormalizedUnsignedByte4>;
#endif // VCL_GRAPHICS_OPENGL_RENDERTYPETRAITS_INST
}}}
#endif // VCL_OPENGL_SUPPORT
