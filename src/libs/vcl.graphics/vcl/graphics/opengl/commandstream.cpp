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

namespace Vcl { namespace Graphics { namespace OpenGL {
	void CommandStream::bind()
	{
		for (auto tok = _commands.begin(); tok != _commands.end(); ++tok)
		{
			switch ((CommandType)*tok)
			{
			case CommandType::Enable:
			{
				const auto cap = (GLenum) * ++tok;
				glEnable(cap);
				break;
			}
			case CommandType::Disable:
			{
				const auto cap = (GLenum) * ++tok;
				glDisable(cap);
				break;
			}
			case CommandType::Enablei:
			{
				const auto cap = (GLenum) * ++tok;
				const auto idx = (GLuint) * ++tok;
				glEnablei(cap, idx);
				break;
			}
			case CommandType::Disablei:
			{
				const auto cap = (GLenum) * ++tok;
				const auto idx = (GLuint) * ++tok;
				glDisablei(cap, idx);
				break;
			}
			case CommandType::LogicOp:
			{
				const auto op = (GLenum) * ++tok;
				glLogicOp(op);
				break;
			}
			case CommandType::BlendFunc:
			{
				const auto sfactor = (GLenum) * ++tok;
				const auto dfactor = (GLenum) * ++tok;
				glBlendFunc(sfactor, dfactor);
				break;
			}
			case CommandType::BlendEquation:
			{
				const auto mode = (GLenum) * ++tok;
				glBlendEquation(mode);
				break;
			}
			case CommandType::BlendFunci:
			{
				const auto buf = (GLuint) * ++tok;
				const auto src = (GLenum) * ++tok;
				const auto dst = (GLenum) * ++tok;
				glBlendFunci(buf, src, dst);
				break;
			}
			case CommandType::BlendEquationi:
			{
				const auto buf = (GLuint) * ++tok;
				const auto mode = (GLenum) * ++tok;
				glBlendEquationi(buf, mode);
				break;
			}
			case CommandType::BlendColor:
			{
				const auto r = toFloat(*++tok);
				const auto g = toFloat(*++tok);
				const auto b = toFloat(*++tok);
				const auto a = toFloat(*++tok);

				glBlendColor(r, g, b, a);
				break;
			}
			case CommandType::ColorMask:
			{
				const auto r = (GLboolean) * ++tok;
				const auto g = (GLboolean) * ++tok;
				const auto b = (GLboolean) * ++tok;
				const auto a = (GLboolean) * ++tok;
				glColorMask(r, g, b, a);
				break;
			}
			case CommandType::ColorMaskIndexed:
			{
				const auto buf = (GLuint) * ++tok;
				const auto r = (GLboolean) * ++tok;
				const auto g = (GLboolean) * ++tok;
				const auto b = (GLboolean) * ++tok;
				const auto a = (GLboolean) * ++tok;
				glColorMaski(buf, r, g, b, a);
				break;
			}
			case CommandType::DepthMask:
			{
				const auto flag = (GLboolean) * ++tok;
				glDepthMask(flag);
				break;
			}
			case CommandType::DepthFunc:
			{
				const auto func = (GLenum) * ++tok;
				glDepthFunc(func);
				break;
			}
			case CommandType::StencilOpSeparate:
			{
				const auto face = (GLenum) * ++tok;
				const auto sfail = (GLenum) * ++tok;
				const auto dpfail = (GLenum) * ++tok;
				const auto dppass = (GLenum) * ++tok;
				glStencilOpSeparate(face, sfail, dpfail, dppass);
				break;
			}
			case CommandType::StencilMaskSeparate:
			{
				const auto face = (GLenum) * ++tok;
				const auto mask = (GLuint) * ++tok;
				glStencilMaskSeparate(face, mask);
				break;
			}
			case CommandType::CullFace:
			{
				const auto mode = (GLenum) * ++tok;
				glCullFace(mode);
				break;
			}
			case CommandType::FrontFace:
			{
				const auto mode = (GLenum) * ++tok;
				glFrontFace(mode);
				break;
			}
			case CommandType::PolygonMode:
			{
				const auto face = (GLenum) * ++tok;
				const auto mode = (GLenum) * ++tok;
				glPolygonMode(face, mode);
				break;
			}
			case CommandType::PolygonOffsetClamp:
			{
				//glPolygonOffsetClampEXT();
				break;
			}
			case CommandType::PolygonOffset:
			{
				//glPolygonOffset();
				break;
			}
			case CommandType::PatchParameteri:
			{
				const auto pname = (GLenum) * ++tok;
				const auto value = (GLint) * ++tok;
				glPatchParameteri(pname, value);
				break;
			}
			case CommandType::DrawArraysInstancedBaseInstance:
			{
				const auto mode = (GLenum) * ++tok;
				const auto first = (GLint) * ++tok;
				const auto count = (GLsizei) * ++tok;
				const auto primcount = (GLsizei) * ++tok;
				const auto baseinstance = (GLuint) * ++tok;
				glDrawArraysInstancedBaseInstance(mode, first, count, primcount, baseinstance);
			}
			case CommandType::DrawElementsInstancedBaseInstance:
			{
				const auto mode = (GLenum) * ++tok;
				const auto count = (GLsizei) * ++tok;
				const auto type = (GLenum) * ++tok;
				const auto indices = (const void*)*++tok;
				const auto primcount = (GLsizei) * ++tok;
				const auto baseinstance = (GLuint) * ++tok;
				glDrawElementsInstancedBaseInstance(mode, count, type, indices, primcount, baseinstance);
				break;
			}
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

			_commands.push_back((uint32_t)type);
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
		return *(uint32_t*)&arg;
	}
	uint32_t CommandStream::toToken(GLenum arg)
	{
		return (uint32_t)arg;
	}
	float CommandStream::toFloat(uint32_t tok)
	{
		return *(float*)&tok;
	}
}}}
