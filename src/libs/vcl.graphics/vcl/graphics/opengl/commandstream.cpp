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
#include <vcl/graphics/opengl/commandstream.h>

// C++ standard library
#include <iterator>

// VCL
#include <vcl/core/contract.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace OpenGL
{
	void CommandStream::bind()
	{
		for (auto tok = _commands.begin(); tok != _commands.end(); ++tok)
		{
			switch ((CommandType) *tok)
			{
			case CommandType::Enable:
				glEnable((GLenum) *++tok);
				break;
			case CommandType::Disable:
				glDisable((GLenum) *++tok);
				break;
			case CommandType::Enablei:
				glEnablei((GLenum) *++tok, (GLuint) *++tok);
				break;
			case CommandType::Disablei:
				glDisablei((GLenum) *++tok, (GLuint) *++tok);
				break;
			case CommandType::LogicOp:
				glLogicOp((GLenum) *++tok);
				break;
			case CommandType::BlendFunc:
				glBlendFunc((GLenum) *++tok, (GLenum) *++tok);
				break;
			case CommandType::BlendEquation:
				glBlendEquation((GLenum) *++tok);
				break;
			case CommandType::BlendFunci:
				glBlendFunci((GLuint) *++tok, (GLenum) *++tok, (GLenum) *++tok);
				break;
			case CommandType::BlendEquationi:
				glBlendEquationi((GLuint) *++tok, (GLenum) *++tok);
				break;
			case CommandType::BlendColor:
				glBlendColor
				(
					toFloat(*++tok),
					toFloat(*++tok),
					toFloat(*++tok),
					toFloat(*++tok)
				);
				break;
			case CommandType::ColorMask:
				glColorMask((GLboolean)*++tok, (GLboolean)*++tok, (GLboolean)*++tok, (GLboolean)*++tok);
				break;
			case CommandType::ColorMaskIndexed:
				glColorMaski((GLuint)*++tok, (GLboolean)*++tok, (GLboolean)*++tok, (GLboolean)*++tok, (GLboolean)*++tok);
				break;
			case CommandType::DepthMask:
				glDepthMask((GLboolean) *++tok);
				break;
			case CommandType::DepthFunc:
				glDepthFunc((GLenum) *++tok);
				break;
			case CommandType::StencilOpSeparate:
				glStencilOpSeparate((GLenum) *++tok, (GLenum) *++tok, (GLenum) *++tok, (GLenum) *++tok);
				break;
			case CommandType::StencilMaskSeparate:
				glStencilMaskSeparate((GLenum) *++tok, (GLuint) *++tok);
				break;
			case CommandType::CullFace:
				glCullFace((GLenum) *++tok);
				break;
			case CommandType::FrontFace:
				glFrontFace((GLenum) *++tok);
				break;
			case CommandType::PolygonMode:
				glPolygonMode((GLenum) *++tok, (GLenum) *++tok);
				break;
			case CommandType::PolygonOffsetClamp:
				//glPolygonOffsetClampEXT();
				break;
			case CommandType::PolygonOffset:
				//glPolygonOffset();
				break;
			case CommandType::PatchParameteri:
				glPatchParameteri((GLenum) *++tok, (GLint) *++tok);
				break;
			case CommandType::DrawArraysInstancedBaseInstance:
				glDrawArraysInstancedBaseInstance((GLenum) *++tok,
					(GLint) *++tok, (GLsizei) *++tok, (GLsizei) *++tok, (GLuint) *++tok);
			case CommandType::DrawElementsInstancedBaseInstance:
				glDrawElementsInstancedBaseInstance((GLenum) *++tok,
					(GLsizei) *++tok, (GLenum) *++tok, (const void*) *++tok, (GLsizei) *++tok, (GLuint) *++tok);
				break;
			}
		}
	}

	void CommandStream::addTokens(CommandType type, std::initializer_list<uint32_t> params)
	{
		auto p = params.begin();

		switch (type)
		{
		case CommandType::Enable:
		case CommandType::Disable:
		case CommandType::BlendEquation:
		case CommandType::DepthMask:
		case CommandType::DepthFunc:
		case CommandType::CullFace:
		case CommandType::FrontFace:
		case CommandType::LogicOp:
		case CommandType::PatchParameteri:
			VclCheck(params.size() == 1, "Number params is valid.");

			_commands.push_back((uint32_t) type);
			_commands.push_back(*p);

			break;
		case CommandType::Enablei:
		case CommandType::Disablei:
		case CommandType::BlendFunc:
		case CommandType::BlendEquationi:
		case CommandType::StencilMaskSeparate:
		case CommandType::PolygonMode:
			VclCheck(params.size() == 2, "Number params is valid.");

			_commands.push_back((uint32_t)type);
			_commands.push_back(*p);
			_commands.push_back(*++p);

			break;
		case CommandType::BlendFunci:
			VclCheck(params.size() == 3, "Number params is valid.");

			_commands.push_back((uint32_t)type);
			_commands.push_back(*p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);

			break;
		case CommandType::BlendColor:
		case CommandType::StencilOpSeparate:
		case CommandType::ColorMask:
			VclCheck(params.size() == 4, "Number params is valid.");

			_commands.push_back((uint32_t)type);
			_commands.push_back(*p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);

			break;
		case CommandType::ColorMaskIndexed:
		case CommandType::DrawArraysInstancedBaseInstance:
			VclCheck(params.size() == 5, "Number params is valid.");

			_commands.push_back((uint32_t)type);
			_commands.push_back(*p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);
			break;
		case CommandType::DrawElementsInstancedBaseInstance:
			VclCheck(params.size() == 6, "Number params is valid.");

			_commands.push_back((uint32_t)type);
			_commands.push_back(*p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);
			_commands.push_back(*++p);
			break;
		case CommandType::PolygonOffsetClamp:
			//glPolygonOffsetClampEXT();
			break;
		case CommandType::PolygonOffset:
			//glPolygonOffset();
			break;
		}
	}

	void CommandStream::emplace(CommandType type, const BindVertexBuffersConfig& config)
	{
		auto stream = reinterpret_cast<const uint32_t*>(&config);
		std::copy(stream, stream + sizeof(config) / sizeof(uint32_t), std::back_inserter(_commands));
	}

	uint32_t CommandStream::toToken(int arg)
	{
		return (uint32_t)arg;
	}
	uint32_t CommandStream::toToken(float arg)
	{
		return *(uint32_t*) &arg;
	}
	uint32_t CommandStream::toToken(GLenum arg)
	{
		return (uint32_t)arg;
	}
	float CommandStream::toFloat(uint32_t tok)
	{
		return *(float*) &tok;
	}
}}}

#endif // VCL_OPENGL_SUPPORT
