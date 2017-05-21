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
#include <vcl/graphics/runtime/opengl/state/sampler.h>

// VCL libraries
#include <vcl/core/contract.h>

#ifdef VCL_OPENGL_SUPPORT
namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Sampler::Sampler(const SamplerDescription& desc)
	: Runtime::Sampler(desc)
	{
		glGenSamplers(1, &_glId);
		
		// Set GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R
		glSamplerParameteri(_glId, GL_TEXTURE_WRAP_S, convert(desc.AddressU));
		glSamplerParameteri(_glId, GL_TEXTURE_WRAP_T, convert(desc.AddressV));
		glSamplerParameteri(_glId, GL_TEXTURE_WRAP_R, convert(desc.AddressW));
		
        // Set GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MAX_ANISOTROPY_EXT
		GLenum min; GLenum mag; GLenum compare_mode;
		convert(desc.Filter, true, min, mag, compare_mode);

		glSamplerParameteri(_glId, GL_TEXTURE_MIN_FILTER, min);
		glSamplerParameteri(_glId, GL_TEXTURE_MAG_FILTER, mag);

		if (desc.Filter == FilterType::Anisotropic || desc.Filter == FilterType::ComparisonAnisotropic)
			glSamplerParameterf(_glId, GL_TEXTURE_MAX_ANISOTROPY_EXT, (GLfloat) desc.MaxAnisotropy);
		else
			glSamplerParameterf(_glId, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1);

        // Set GL_TEXTURE_MIN_LOD, GL_TEXTURE_MAX_LOD, GL_TEXTURE_LOD_BIAS
		glSamplerParameterf(_glId, GL_TEXTURE_MIN_LOD, desc.MinLOD);
		glSamplerParameterf(_glId, GL_TEXTURE_MAX_LOD, desc.MaxLOD);
		glSamplerParameterf(_glId, GL_TEXTURE_LOD_BIAS, desc.MipLODBias);

		// Set GL_TEXTURE_BORDER_COLOR
		glSamplerParameterfv(_glId, GL_TEXTURE_BORDER_COLOR, desc.BorderColor.data());

        // Set GL_TEXTURE_COMPARE_MODE, GL_TEXTURE_COMPARE_FUNC
		glSamplerParameteri(_glId, GL_TEXTURE_COMPARE_MODE, compare_mode);
		glSamplerParameteri(_glId, GL_TEXTURE_COMPARE_FUNC, convert(desc.ComparisonFunc));

		VclEnsure(_glId > 0, "Sampler is created.");
	}

	Sampler::~Sampler()
	{
		glDeleteSamplers(1, &_glId);
	}
	
	void Sampler::convert(FilterType filter, bool enable_mipmap, GLenum& min, GLenum& mag, GLenum& compare_mode) const
	{
		if (enable_mipmap)
		{
			switch (filter)
			{
			case FilterType::MinMagMipPoint                      : min = GL_NEAREST_MIPMAP_NEAREST; mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::MinMagPointMipLinear                : min = GL_NEAREST_MIPMAP_LINEAR;  mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::MinPointMagLinearMipPoint           : min = GL_NEAREST_MIPMAP_NEAREST; mag = GL_LINEAR;  compare_mode = GL_NONE; return;
			case FilterType::MinPointMagMipLinear                : min = GL_NEAREST_MIPMAP_LINEAR;  mag = GL_LINEAR;  compare_mode = GL_NONE; return;
			case FilterType::MinLinearMagMipPoint                : min = GL_LINEAR_MIPMAP_NEAREST;  mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::MinLinearMagPointMipLinear          : min = GL_LINEAR_MIPMAP_LINEAR;   mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::MinMagLinearMipPoint                : min = GL_LINEAR_MIPMAP_NEAREST;  mag = GL_LINEAR;  compare_mode = GL_NONE; return;
			case FilterType::MinMagMipLinear                     : min = GL_LINEAR_MIPMAP_LINEAR;   mag = GL_LINEAR;  compare_mode = GL_NONE; return;
			case FilterType::Anisotropic                         : min = GL_NEAREST_MIPMAP_NEAREST; mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::ComparisonMinMagMipPoint            : min = GL_NEAREST_MIPMAP_NEAREST; mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinMagPointMipLinear      : min = GL_NEAREST_MIPMAP_LINEAR;  mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinPointMagLinearMipPoint : min = GL_NEAREST_MIPMAP_NEAREST; mag = GL_LINEAR;  compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinPointMagMipLinear      : min = GL_NEAREST_MIPMAP_LINEAR;  mag = GL_LINEAR;  compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinLinearMagMipPoint      : min = GL_LINEAR_MIPMAP_NEAREST;  mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinLinearMagPointMipLinear: min = GL_LINEAR_MIPMAP_LINEAR;   mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinMagLinearMipPoint      : min = GL_LINEAR_MIPMAP_NEAREST;  mag = GL_LINEAR;  compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinMagMipLinear           : min = GL_LINEAR_MIPMAP_LINEAR;   mag = GL_LINEAR;  compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonAnisotropic               : min = GL_NEAREST_MIPMAP_NEAREST; mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			default: { VclDebugError("Enumeration value is valid."); }
			}
		}
		else
		{
			switch (filter)
			{
			case FilterType::MinMagMipPoint                      : min = GL_NEAREST; mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::MinMagPointMipLinear                : min = GL_NEAREST; mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::MinPointMagLinearMipPoint           : min = GL_NEAREST; mag = GL_LINEAR;  compare_mode = GL_NONE; return;
			case FilterType::MinPointMagMipLinear                : min = GL_NEAREST; mag = GL_LINEAR;  compare_mode = GL_NONE; return;
			case FilterType::MinLinearMagMipPoint                : min = GL_LINEAR;  mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::MinLinearMagPointMipLinear          : min = GL_LINEAR;  mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::MinMagLinearMipPoint                : min = GL_LINEAR;  mag = GL_LINEAR;  compare_mode = GL_NONE; return;
			case FilterType::MinMagMipLinear                     : min = GL_LINEAR;  mag = GL_LINEAR;  compare_mode = GL_NONE; return;
			case FilterType::Anisotropic                         : min = GL_NEAREST; mag = GL_NEAREST; compare_mode = GL_NONE; return;
			case FilterType::ComparisonMinMagMipPoint            : min = GL_NEAREST; mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinMagPointMipLinear      : min = GL_NEAREST; mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinPointMagLinearMipPoint : min = GL_NEAREST; mag = GL_LINEAR;  compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinPointMagMipLinear      : min = GL_NEAREST; mag = GL_LINEAR;  compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinLinearMagMipPoint      : min = GL_LINEAR;  mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinLinearMagPointMipLinear: min = GL_LINEAR;  mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinMagLinearMipPoint      : min = GL_LINEAR;  mag = GL_LINEAR;  compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonMinMagMipLinear           : min = GL_LINEAR;  mag = GL_LINEAR;  compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			case FilterType::ComparisonAnisotropic               : min = GL_NEAREST; mag = GL_NEAREST; compare_mode = GL_COMPARE_REF_TO_TEXTURE; return;
			default: { VclDebugError("Enumeration value is valid."); }
			}
		}
	}

	GLenum Sampler::convert(TextureAddressMode mode) const
	{
		VCL_WARNING("Check texture address mode conversion for Clamp and MirrorOnce.")

		switch (mode)
		{
		case TextureAddressMode::Wrap      : return GL_REPEAT;
		case TextureAddressMode::Mirror    : return GL_MIRRORED_REPEAT;
		case TextureAddressMode::Clamp     : return GL_CLAMP_TO_EDGE;
		case TextureAddressMode::Border    : return GL_CLAMP_TO_BORDER;
		case TextureAddressMode::MirrorOnce: { VclDebugError("Not supported."); break; }
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_NONE;
	}

	GLenum Sampler::convert(ComparisonFunction func) const
	{
		switch (func)
		{
		case ComparisonFunction::Never       : return GL_NEVER;
		case ComparisonFunction::Less        : return GL_LESS;
		case ComparisonFunction::Equal       : return GL_EQUAL;
		case ComparisonFunction::LessEqual   : return GL_LEQUAL;
		case ComparisonFunction::Greater     : return GL_GREATER;
		case ComparisonFunction::NotEqual    : return GL_NOTEQUAL;
		case ComparisonFunction::GreaterEqual: return GL_GEQUAL;
		case ComparisonFunction::Always      : return GL_ALWAYS;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_NONE;
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
