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

#define VCL_GRAPHICS_RTT(type, int_fmt, fmt, cmp_type, nr_cmp, integral)	   \
	template<>																   \
	struct RenderTypeTrait<type>											   \
	{																		   \
		typedef type Type;													   \
		static const GLenum InternalFormat;									   \
		static const GLenum Format;											   \
		static const GLenum ComponentType;									   \
		static const GLint ComponentSize;									   \
		static const GLint NrComponents;									   \
		static const GLint Size;											   \
		static const bool IsIntegral;										   \
	};																		   \
	const GLenum RenderTypeTrait<type>::InternalFormat = int_fmt;			   \
	const GLenum RenderTypeTrait<type>::Format = fmt;						   \
	const GLenum RenderTypeTrait<type>::ComponentType = cmp_type;			   \
	const GLint  RenderTypeTrait<type>::ComponentSize = sizeof(type) / nr_cmp; \
	const GLint  RenderTypeTrait<type>::NrComponents = nr_cmp;				   \
	const GLint  RenderTypeTrait<type>::Size = sizeof(type);				   \
	const bool   RenderTypeTrait<type>::IsIntegral = integral

namespace Vcl { namespace Graphics { namespace OpenGL
{
	GLenum AnyRenderType::internalFormat() const  { return _gate().internalFormat(); }
	GLenum AnyRenderType::format() const  { return _gate().format(); }
	GLenum AnyRenderType::componentType() const { return _gate().componentType(); }
	GLint AnyRenderType::size() const { return _gate().size(); }
	GLint AnyRenderType::componentSize() const { return _gate().componentSize(); }
	GLint AnyRenderType::nrComponents() const { return _gate().nrComponents(); }
	bool AnyRenderType::isIntegral() const { return _gate().isIntegral(); }

	template<>
	struct RenderTypeTrait<void>
	{
		typedef void Type;
		static const GLenum InternalFormat;
		static const GLenum Format;
		static const GLenum ComponentType;
		static const GLint ComponentSize;
		static const GLint NrComponents;
		static const GLint Size;
		static const bool IsIntegral;
	};
	const GLenum RenderTypeTrait<void>::InternalFormat = GL_NONE;
	const GLenum RenderTypeTrait<void>::Format = GL_NONE;
	const GLenum RenderTypeTrait<void>::ComponentType = GL_NONE;
	const GLint  RenderTypeTrait<void>::ComponentSize = 1;
	const GLint  RenderTypeTrait<void>::NrComponents = 1;
	const GLint  RenderTypeTrait<void>::Size = 1;
	const bool   RenderTypeTrait<void>::IsIntegral = false;
	
	VCL_GRAPHICS_RTT(float,           GL_R32F,    GL_RED,  GL_FLOAT, 1, false);
	VCL_GRAPHICS_RTT(Eigen::Vector2f, GL_RG32F,   GL_RG,   GL_FLOAT, 2, false);
	VCL_GRAPHICS_RTT(Eigen::Vector3f, GL_RGB32F,  GL_RGB,  GL_FLOAT, 3, false);
	VCL_GRAPHICS_RTT(Eigen::Vector4f, GL_RGBA32F, GL_RGBA, GL_FLOAT, 4, false);

	VCL_GRAPHICS_RTT(int,             GL_R32I,    GL_RED_INTEGER,  GL_INT, 1, true);
	VCL_GRAPHICS_RTT(Eigen::Vector2i, GL_RG32I,   GL_RG_INTEGER,   GL_INT, 2, true);
	VCL_GRAPHICS_RTT(Eigen::Vector3i, GL_RGB32I,  GL_RGB_INTEGER,  GL_INT, 3, true);
	VCL_GRAPHICS_RTT(Eigen::Vector4i, GL_RGBA32I, GL_RGBA_INTEGER, GL_INT, 4, true);

	VCL_GRAPHICS_RTT(unsigned int,     GL_R32UI,    GL_RED_INTEGER,  GL_UNSIGNED_INT, 1, true);
	VCL_GRAPHICS_RTT(Eigen::Vector2ui, GL_RG32UI,   GL_RG_INTEGER,   GL_UNSIGNED_INT, 2, true);
	VCL_GRAPHICS_RTT(Eigen::Vector3ui, GL_RGB32UI,  GL_RGB_INTEGER,  GL_UNSIGNED_INT, 3, true);
	VCL_GRAPHICS_RTT(Eigen::Vector4ui, GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 4, true);
	
	VCL_GRAPHICS_RTT(Float , GL_R32F,    GL_RED,  GL_FLOAT, 1, false);
	VCL_GRAPHICS_RTT(Float2, GL_RG32F,   GL_RG,   GL_FLOAT, 2, false);
	VCL_GRAPHICS_RTT(Float3, GL_RGB32F,  GL_RGB,  GL_FLOAT, 3, false);
	VCL_GRAPHICS_RTT(Float4, GL_RGBA32F, GL_RGBA, GL_FLOAT, 4, false);

	VCL_GRAPHICS_RTT(SignedInt,  GL_R32I,    GL_RED_INTEGER,  GL_INT, 1, true);
	VCL_GRAPHICS_RTT(SignedInt2, GL_RG32I,   GL_RG_INTEGER,   GL_INT, 2, true);
	VCL_GRAPHICS_RTT(SignedInt3, GL_RGB32I,  GL_RGB_INTEGER,  GL_INT, 3, true);
	VCL_GRAPHICS_RTT(SignedInt4, GL_RGBA32I, GL_RGBA_INTEGER, GL_INT, 4, true);

	VCL_GRAPHICS_RTT(UnsignedInt,  GL_R32UI,    GL_RED_INTEGER,  GL_UNSIGNED_INT, 1, true);
	VCL_GRAPHICS_RTT(UnsignedInt2, GL_RG32UI,   GL_RG_INTEGER,   GL_UNSIGNED_INT, 2, true);
	VCL_GRAPHICS_RTT(UnsignedInt3, GL_RGB32UI,  GL_RGB_INTEGER,  GL_UNSIGNED_INT, 3, true);
	VCL_GRAPHICS_RTT(UnsignedInt4, GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 4, true);
	
	VCL_GRAPHICS_RTT(Half , GL_R16F,    GL_RED,  GL_HALF_FLOAT, 1, false);
	VCL_GRAPHICS_RTT(Half2, GL_RG16F,   GL_RG,   GL_HALF_FLOAT, 2, false);
	VCL_GRAPHICS_RTT(Half3, GL_RGB16F,  GL_RGB,  GL_HALF_FLOAT, 3, false);
	VCL_GRAPHICS_RTT(Half4, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 4, false);

	VCL_GRAPHICS_RTT(short,        GL_R16I,    GL_RED_INTEGER,  GL_SHORT, 1, true);
	VCL_GRAPHICS_RTT(SignedShort,  GL_R16I,    GL_RED_INTEGER,  GL_SHORT, 1, true);
	VCL_GRAPHICS_RTT(SignedShort2, GL_RG16I,   GL_RG_INTEGER,   GL_SHORT, 2, true);
	VCL_GRAPHICS_RTT(SignedShort3, GL_RGB16I,  GL_RGB_INTEGER,  GL_SHORT, 3, true);
	VCL_GRAPHICS_RTT(SignedShort4, GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT, 4, true);
	VCL_GRAPHICS_RTT(NormalizedSignedShort,  GL_R16I,    GL_RED,  GL_SHORT, 1, false);
	VCL_GRAPHICS_RTT(NormalizedSignedShort2, GL_RG16I,   GL_RG,   GL_SHORT, 2, false);
	VCL_GRAPHICS_RTT(NormalizedSignedShort3, GL_RGB16I,  GL_RGB,  GL_SHORT, 3, false);
	VCL_GRAPHICS_RTT(NormalizedSignedShort4, GL_RGBA16I, GL_RGBA, GL_SHORT, 4, false);

	VCL_GRAPHICS_RTT(unsigned short, GL_R16UI,    GL_RED_INTEGER,  GL_UNSIGNED_SHORT, 1, true);
	VCL_GRAPHICS_RTT(UnsignedShort,  GL_R16UI,    GL_RED_INTEGER,  GL_UNSIGNED_SHORT, 1, true);
	VCL_GRAPHICS_RTT(UnsignedShort2, GL_RG16UI,   GL_RG_INTEGER,   GL_UNSIGNED_SHORT, 2, true);
	VCL_GRAPHICS_RTT(UnsignedShort3, GL_RGB16UI,  GL_RGB_INTEGER,  GL_UNSIGNED_SHORT, 3, true);
	VCL_GRAPHICS_RTT(UnsignedShort4, GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, 4, true);
	VCL_GRAPHICS_RTT(NormalizedUnsignedShort,  GL_R16UI,    GL_RED,  GL_UNSIGNED_SHORT, 1, false);
	VCL_GRAPHICS_RTT(NormalizedUnsignedShort2, GL_RG16UI,   GL_RG,   GL_UNSIGNED_SHORT, 2, false);
	VCL_GRAPHICS_RTT(NormalizedUnsignedShort3, GL_RGB16UI,  GL_RGB,  GL_UNSIGNED_SHORT, 3, false);
	VCL_GRAPHICS_RTT(NormalizedUnsignedShort4, GL_RGBA16UI, GL_RGBA, GL_UNSIGNED_SHORT, 4, false);
	
	VCL_GRAPHICS_RTT(char,        GL_R8I,    GL_RED_INTEGER,  GL_BYTE, 1, true);
	VCL_GRAPHICS_RTT(SignedByte,  GL_R8I,    GL_RED_INTEGER,  GL_BYTE, 1, true);
	VCL_GRAPHICS_RTT(SignedByte2, GL_RG8I,   GL_RG_INTEGER,   GL_BYTE, 2, true);
	VCL_GRAPHICS_RTT(SignedByte3, GL_RGB8I,  GL_RGB_INTEGER,  GL_BYTE, 3, true);
	VCL_GRAPHICS_RTT(SignedByte4, GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE, 4, true);
	VCL_GRAPHICS_RTT(NormalizedSignedByte,  GL_R8I,    GL_RED,  GL_BYTE, 1, false);
	VCL_GRAPHICS_RTT(NormalizedSignedByte2, GL_RG8I,   GL_RG,   GL_BYTE, 2, false);
	VCL_GRAPHICS_RTT(NormalizedSignedByte3, GL_RGB8I,  GL_RGB,  GL_BYTE, 3, false);
	VCL_GRAPHICS_RTT(NormalizedSignedByte4, GL_RGBA8I, GL_RGBA, GL_BYTE, 4, false);

	VCL_GRAPHICS_RTT(unsigned char, GL_R8UI,    GL_RED_INTEGER,  GL_UNSIGNED_BYTE, 1, true);
	VCL_GRAPHICS_RTT(UnsignedByte,  GL_R8UI,    GL_RED_INTEGER,  GL_UNSIGNED_BYTE, 1, true);
	VCL_GRAPHICS_RTT(UnsignedByte2, GL_RG8UI,   GL_RG_INTEGER,   GL_UNSIGNED_BYTE, 2, true);
	VCL_GRAPHICS_RTT(UnsignedByte3, GL_RGB8UI,  GL_RGB_INTEGER,  GL_UNSIGNED_BYTE, 3, true);
	VCL_GRAPHICS_RTT(UnsignedByte4, GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, 4, true);
	VCL_GRAPHICS_RTT(NormalizedUnsignedByte,  GL_R8UI,    GL_RED,  GL_UNSIGNED_BYTE, 1, false);
	VCL_GRAPHICS_RTT(NormalizedUnsignedByte2, GL_RG8UI,   GL_RG,   GL_UNSIGNED_BYTE, 2, false);
	VCL_GRAPHICS_RTT(NormalizedUnsignedByte3, GL_RGB8UI,  GL_RGB,  GL_UNSIGNED_BYTE, 3, false);
	VCL_GRAPHICS_RTT(NormalizedUnsignedByte4, GL_RGBA8UI, GL_RGBA, GL_UNSIGNED_BYTE, 4, false);
}}}
#endif // VCL_OPENGL_SUPPORT
