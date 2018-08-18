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
#include <vcl/graphics/runtime/opengl/state/blendstate.h>

// C++ standard library
#include <utility>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/opengl/gl.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	using namespace Vcl::Graphics::OpenGL;

	BlendState::BlendState(const BlendDescription& desc)
	: _desc(desc)
	{
		// Check consistency
		VclRequire(implies(desc.IndependentBlendEnable, glewIsSupported("GL_ARB_draw_buffers_blend") && glewIsSupported("GL_EXT_draw_buffers2")), "Independent blending is supported.");
		VclRequire(implies(desc.LogicOpEnable, desc.RenderTarget[0].BlendEnable == false && desc.IndependentBlendEnable == false), "Either logic ops or blend ops are enabled.");
		VclRequire(implies(desc.RenderTarget[0].BlendEnable, desc.LogicOpEnable == false), "Either logic ops or blend ops are enabled.");

		VclRequire(implies(desc.RenderTarget[0].BlendOp >= BlendOperation::Multiply, glewIsSupported("GL_KHR_blend_equation_advanced")), "Advanced blending operations are supported.");
		VclRequire(implies(desc.RenderTarget[1].BlendOp >= BlendOperation::Multiply, glewIsSupported("GL_KHR_blend_equation_advanced")), "Advanced blending operations are supported.");
		VclRequire(implies(desc.RenderTarget[2].BlendOp >= BlendOperation::Multiply, glewIsSupported("GL_KHR_blend_equation_advanced")), "Advanced blending operations are supported.");
		VclRequire(implies(desc.RenderTarget[3].BlendOp >= BlendOperation::Multiply, glewIsSupported("GL_KHR_blend_equation_advanced")), "Advanced blending operations are supported.");
		VclRequire(implies(desc.RenderTarget[4].BlendOp >= BlendOperation::Multiply, glewIsSupported("GL_KHR_blend_equation_advanced")), "Advanced blending operations are supported.");
		VclRequire(implies(desc.RenderTarget[5].BlendOp >= BlendOperation::Multiply, glewIsSupported("GL_KHR_blend_equation_advanced")), "Advanced blending operations are supported.");
		VclRequire(implies(desc.RenderTarget[6].BlendOp >= BlendOperation::Multiply, glewIsSupported("GL_KHR_blend_equation_advanced")), "Advanced blending operations are supported.");
		VclRequire(implies(desc.RenderTarget[7].BlendOp >= BlendOperation::Multiply, glewIsSupported("GL_KHR_blend_equation_advanced")), "Advanced blending operations are supported.");
	}

	bool BlendState::isIndependentBlendingSupported()
	{
		return glewIsSupported("GL_ARB_draw_buffers_blend") != 0 && glewIsSupported("GL_EXT_draw_buffers2") != 0;
	}
	bool BlendState::areAdvancedBlendOperationsSupported()
	{
		return glewIsSupported("GL_KHR_blend_equation_advanced") != 0;
	}

	void BlendState::bind()
	{
		// Alpha to coverage
		if (desc().AlphaToCoverageEnable)
		{
			glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
		}
		else
		{
			glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
		}

		if (desc().LogicOpEnable && desc().RenderTarget[0].BlendEnable == false)
		{
			glEnable(GL_COLOR_LOGIC_OP);
			glLogicOp(toGLenum(desc().LogicOp));

			// Disable blending when logic operations are used
			glDisable(GL_BLEND);
		}
		else
		{
			// Disable logic operations
			glDisable(GL_COLOR_LOGIC_OP);
		}

		if (desc().IndependentBlendEnable == false)
		{
			const auto& rt = desc().RenderTarget[0];
			if (rt.BlendEnable && desc().LogicOpEnable == false)
			{
				if (rt.SrcBlend == Blend::BlendFactor    || 
					rt.SrcBlend == Blend::InvBlendFactor || 
					rt.DestBlend == Blend::BlendFactor   ||
					rt.DestBlend == Blend::InvBlendFactor  )
				{
					float r = desc().ConstantColor[0];
					float g = desc().ConstantColor[1];
					float b = desc().ConstantColor[2];
					float a = desc().ConstantColor[3];

					glBlendColor(r, g, b, a);
				}

				glEnable(GL_BLEND);
				glBlendFunc(toGLenum(rt.SrcBlend), toGLenum(rt.DestBlend));
				glBlendEquation(toGLenum(rt.BlendOp));
			}
			else
			{
				glDisable(GL_BLEND);
			}

			// Set write mask
			glColorMask
			(
				rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Red),
				rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Green),
				rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Blue),
				rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Alpha)
			);
		}
		else
		{
			GLint i = 0;
			for (const auto& rt : desc().RenderTarget)
			{
				if (rt.BlendEnable && desc().LogicOpEnable == false)
				{
					glEnablei(GL_BLEND, i);
					glBlendFunci(i, toGLenum(rt.SrcBlend), toGLenum(rt.DestBlend));
					glBlendEquationi(i, toGLenum(rt.BlendOp));
				}
				else
				{
					glDisablei(GL_BLEND, i);
				}

				// Set write mask
				glColorMaski
				(
					i,
					rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Red),
					rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Green),
					rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Blue),
					rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Alpha)
				);

				i++;
			}
		}
	}

	void BlendState::record(Graphics::OpenGL::CommandStream& states)
	{
		// Alpha to coverage
		if (desc().AlphaToCoverageEnable)
		{
			states.emplace(CommandType::Enable, GL_SAMPLE_ALPHA_TO_COVERAGE);
		}

		// Enable the logic operations if applicable
		if (desc().LogicOpEnable && desc().RenderTarget[0].BlendEnable == false)
		{
			states.emplace(CommandType::Enable, GL_COLOR_LOGIC_OP);
			states.emplace(CommandType::LogicOp, toGLenum(desc().LogicOp));
		}

		if (desc().IndependentBlendEnable == false)
		{
			const auto& rt = desc().RenderTarget[0];
			if (rt.BlendEnable && desc().LogicOpEnable == false)
			{
				if (rt.SrcBlend == Blend::BlendFactor ||
					rt.SrcBlend == Blend::InvBlendFactor ||
					rt.DestBlend == Blend::BlendFactor ||
					rt.DestBlend == Blend::InvBlendFactor)
				{
					float r = desc().ConstantColor[0];
					float g = desc().ConstantColor[1];
					float b = desc().ConstantColor[2];
					float a = desc().ConstantColor[3];

					states.emplace(CommandType::BlendColor, r, g, b, a);
				}

				states.emplace(CommandType::Enable, GL_BLEND);
				states.emplace(CommandType::BlendFunc, toGLenum(rt.SrcBlend), toGLenum(rt.DestBlend));
				states.emplace(CommandType::BlendEquation, toGLenum(rt.BlendOp));
			}

			// Set write mask
			if (!rt.RenderTargetWriteMask.areAllSet())
			{
				states.emplace
				(
					CommandType::ColorMask,
					rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Red),
					rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Green),
					rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Blue),
					rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Alpha)
				);
			}
		}
		else
		{
			GLint i = 0;
			for (const auto& rt : desc().RenderTarget)
			{
				if (rt.BlendEnable && desc().LogicOpEnable == false)
				{
					states.emplace(CommandType::Enablei, GL_BLEND, i);
					states.emplace(CommandType::BlendFunci, i, toGLenum(rt.SrcBlend), toGLenum(rt.DestBlend));
					states.emplace(CommandType::BlendEquationi, i, toGLenum(rt.BlendOp));
				}

				// Set write mask
				if (!rt.RenderTargetWriteMask.areAllSet())
				{
					states.emplace
					(
						CommandType::ColorMaskIndexed,
						i,
						rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Red),
						rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Green),
						rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Blue),
						rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Alpha)
					);
				}
				i++;
			}
		}
	}

	bool BlendState::isValid() const
	{
		bool valid = true;

		if (desc().LogicOpEnable)
		{
			valid &= glIsEnabled(GL_COLOR_LOGIC_OP) == GL_TRUE;
			valid &= GL::getEnum(GL_LOGIC_OP_MODE) == toGLenum(desc().LogicOp);
		}
		else
		{
			valid &= glIsEnabled(GL_COLOR_LOGIC_OP) == GL_FALSE;
		}

		if (desc().IndependentBlendEnable == false)
		{
			const auto& rt = desc().RenderTarget[0];
			if (rt.BlendEnable && desc().LogicOpEnable == false)
			{
				valid &= glIsEnabled(GL_BLEND) == GL_TRUE;
				valid &= GL::getEnum(GL_BLEND_SRC) == toGLenum(rt.SrcBlend);
				valid &= GL::getEnum(GL_BLEND_DST) == toGLenum(rt.DestBlend);
				valid &= GL::getEnum(GL_BLEND_EQUATION_RGB) == toGLenum(rt.BlendOp);
			}
			else
			{
				valid &= glIsEnabled(GL_BLEND) == GL_FALSE;
			}

			// Check write mask
			GLint values[4];
			glGetIntegerv(GL_COLOR_WRITEMASK, values);
			valid &= values[0] > 0 == (rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Red));
			valid &= values[1] > 0 == (rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Green));
			valid &= values[2] > 0 == (rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Blue));
			valid &= values[3] > 0 == (rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Alpha));
		}
		else
		{
			for (unsigned int i = 0; i < desc().RenderTarget.size(); i++)
			{
				const auto& rt = desc().RenderTarget[i];

				if (rt.BlendEnable && desc().LogicOpEnable == false)
				{
					valid &= glIsEnabledi(GL_BLEND, i) == GL_TRUE;
					valid &= GL::getEnum(GL_BLEND_SRC, i) == toGLenum(desc().RenderTarget[i].SrcBlend);
					valid &= GL::getEnum(GL_BLEND_DST, i) == toGLenum(desc().RenderTarget[i].DestBlend);
					valid &= GL::getEnum(GL_BLEND_EQUATION_RGB, i) == toGLenum(desc().RenderTarget[i].BlendOp);
				}
				else
				{
					valid &= glIsEnabledi(GL_BLEND, i) == GL_FALSE;
				}

				// Check write mask
				GLint values[4];
				glGetIntegeri_v(GL_COLOR_WRITEMASK, i, values);
				valid &= values[0] > 0 == (desc().RenderTarget[i].RenderTargetWriteMask.isSet(ColourWriteEnable::Red));
				valid &= values[1] > 0 == (desc().RenderTarget[i].RenderTargetWriteMask.isSet(ColourWriteEnable::Green));
				valid &= values[2] > 0 == (desc().RenderTarget[i].RenderTargetWriteMask.isSet(ColourWriteEnable::Blue));
				valid &= values[3] > 0 == (desc().RenderTarget[i].RenderTargetWriteMask.isSet(ColourWriteEnable::Alpha));
			}
		}

		return valid;
	}

	bool BlendState::check() const
	{
		if (desc().LogicOpEnable)
		{
			VclCheck(glIsEnabled(GL_COLOR_LOGIC_OP) == GL_TRUE, "Logic Op state is enabled.");
			VclCheck(GL::getEnum(GL_LOGIC_OP_MODE) == toGLenum(desc().LogicOp), "Logic Op mode is correct.");
		}
		else
		{
			VclCheck(glIsEnabled(GL_COLOR_LOGIC_OP) == GL_FALSE, "Logic Op state is disabled.");
		}

		if (desc().IndependentBlendEnable == false)
		{
			const auto& rt = desc().RenderTarget[0];
			if (rt.BlendEnable && desc().LogicOpEnable == false)
			{
				VclCheck(glIsEnabled(GL_BLEND) == GL_TRUE, "Blend state is enabled.");
				VclCheck(GL::getEnum(GL_BLEND_SRC) == toGLenum(rt.SrcBlend), "Src blend is correct.");
				VclCheck(GL::getEnum(GL_BLEND_DST) == toGLenum(rt.DestBlend), "Dest blend is correct.");
				VclCheck(GL::getEnum(GL_BLEND_EQUATION_RGB) == toGLenum(rt.BlendOp), "Blend Equation is correct.");
			}
			else
			{
				VclCheck(glIsEnabled(GL_BLEND) == GL_FALSE, "Blend state is disabled.");
			}

			// Check write mask
			VclAssertBlock
			{
				GLint values[4];
				glGetIntegerv(GL_COLOR_WRITEMASK, values);
				VclCheck(values[0] > 0 == (rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Red)), "Red write mask is correct.");
				VclCheck(values[1] > 0 == (rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Green)), "Green write mask is correct.");
				VclCheck(values[2] > 0 == (rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Blue)), "Blue write mask is correct.");
				VclCheck(values[3] > 0 == (rt.RenderTargetWriteMask.isSet(ColourWriteEnable::Alpha)), "Alpha write mask is correct.");
			}
		}
		else
		{
			for (unsigned int i = 0; i < desc().RenderTarget.size(); i++)
			{
				const auto& rt = desc().RenderTarget[i];

				if (rt.BlendEnable && desc().LogicOpEnable == false)
				{
					VclCheck(glIsEnabledi(GL_BLEND, i) == GL_TRUE, "Blend state is enabled.");
					VclCheck(GL::getEnum(GL_BLEND_SRC, i) == toGLenum(desc().RenderTarget[i].SrcBlend), "Src blend is correct.");
					VclCheck(GL::getEnum(GL_BLEND_DST, i) == toGLenum(desc().RenderTarget[i].DestBlend), "Dest blend is correct.");
					VclCheck(GL::getEnum(GL_BLEND_EQUATION_RGB, i) == toGLenum(desc().RenderTarget[i].BlendOp), "Blend Equation is correct.");
				}
				else
				{
					VclCheck(glIsEnabledi(GL_BLEND, i) == GL_FALSE, "Blend state is disabled.");
				}
				
				// Check write mask
				VclAssertBlock
				{
					GLint values[4];
					glGetIntegeri_v(GL_COLOR_WRITEMASK, i, values);
					VclCheck(values[0] > 0 == (desc().RenderTarget[i].RenderTargetWriteMask.isSet(ColourWriteEnable::Red)), "Red write mask is correct.");
					VclCheck(values[1] > 0 == (desc().RenderTarget[i].RenderTargetWriteMask.isSet(ColourWriteEnable::Green)), "Green write mask is correct.");
					VclCheck(values[2] > 0 == (desc().RenderTarget[i].RenderTargetWriteMask.isSet(ColourWriteEnable::Blue)), "Blue write mask is correct.");
					VclCheck(values[3] > 0 == (desc().RenderTarget[i].RenderTargetWriteMask.isSet(ColourWriteEnable::Alpha)), "Alpha write mask is correct.");
				}
			}
		}
		
		return true;
	}

	GLenum BlendState::toGLenum(LogicOperation op)
	{
		switch (op)
		{
		case LogicOperation::Clear       : return GL_CLEAR;
		case LogicOperation::Set         : return GL_SET;
		case LogicOperation::Copy        : return GL_COPY;
		case LogicOperation::CopyInverted: return GL_COPY_INVERTED;
		case LogicOperation::NoOp        : return GL_NOOP;
		case LogicOperation::Invert      : return GL_INVERT;
		case LogicOperation::And         : return GL_AND;
		case LogicOperation::Nand        : return GL_NAND;
		case LogicOperation::Or          : return GL_OR;
		case LogicOperation::Nor         : return GL_NOR;
		case LogicOperation::Xor         : return GL_XOR;
		case LogicOperation::Equiv       : return GL_EQUIV;
		case LogicOperation::AndReverse  : return GL_AND_REVERSE;
		case LogicOperation::AndInverted : return GL_AND_INVERTED;
		case LogicOperation::OrReverse   : return GL_OR_REVERSE;
		case LogicOperation::OrInverted  : return GL_OR_INVERTED;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_INVALID_ENUM;
	}
	GLenum BlendState::toGLenum(BlendOperation op)
	{
		switch (op)
		{
		case BlendOperation::Add        :  return GL_FUNC_ADD;
		case BlendOperation::Subtract   :  return GL_FUNC_SUBTRACT;
		case BlendOperation::RevSubtract:  return GL_FUNC_REVERSE_SUBTRACT;
		case BlendOperation::Min        :  return GL_MIN;
		case BlendOperation::Max        :  return GL_MAX;

		case BlendOperation::Multiply   : return GL_MULTIPLY_KHR;
		case BlendOperation::Screen     : return GL_SCREEN_KHR;
		case BlendOperation::Overlay    : return GL_OVERLAY_KHR;
		case BlendOperation::Darken     : return GL_DARKEN_KHR;
		case BlendOperation::Lighten    : return GL_LIGHTEN_KHR;
		case BlendOperation::Colordodge : return GL_COLORDODGE_KHR;
		case BlendOperation::Colorburn  : return GL_COLORBURN_KHR;
		case BlendOperation::Hardlight  : return GL_HARDLIGHT_KHR;
		case BlendOperation::Softlight  : return GL_SOFTLIGHT_KHR;
		case BlendOperation::Difference : return GL_DIFFERENCE_KHR;
		case BlendOperation::Exclusion  : return GL_EXCLUSION_KHR;

		case BlendOperation::HslHue        : return GL_HSL_HUE_KHR;
		case BlendOperation::HslSaturation : return GL_HSL_SATURATION_KHR;
		case BlendOperation::HslColor      : return GL_HSL_COLOR_KHR;
		case BlendOperation::HslLuminosity : return GL_HSL_LUMINOSITY_KHR;

		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_INVALID_ENUM;
	}

	GLenum BlendState::toGLenum(Blend factor)
	{
		switch (factor)
		{
		case Blend::Zero          : return GL_ZERO;
		case Blend::One           : return GL_ONE;
		case Blend::SrcColour     : return GL_SRC_COLOR;
		case Blend::InvSrcColour  : return GL_ONE_MINUS_SRC_COLOR;
		case Blend::SrcAlpha      : return GL_SRC_ALPHA;
		case Blend::InvSrcAlpha   : return GL_ONE_MINUS_SRC_ALPHA;
		case Blend::DestAlpha     : return GL_DST_ALPHA;
		case Blend::InvDestAlpha  : return GL_ONE_MINUS_DST_ALPHA;
		case Blend::DestColour    : return GL_DST_COLOR;
		case Blend::InvDestColour : return GL_ONE_MINUS_DST_COLOR;
		case Blend::SrcAlphaSat   : return GL_SRC_ALPHA_SATURATE;
		case Blend::BlendFactor   : return GL_CONSTANT_COLOR;
		case Blend::InvBlendFactor: return GL_ONE_MINUS_CONSTANT_COLOR;
		case Blend::Src1Colour    : return GL_SRC1_COLOR;
		case Blend::InvSrc1Colour : return GL_ONE_MINUS_SRC1_COLOR;
		case Blend::Src1Alpha     : return GL_SRC1_ALPHA;
		case Blend::InvSrc1Alpha  : return GL_ONE_MINUS_SRC1_ALPHA;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return GL_INVALID_ENUM;
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
