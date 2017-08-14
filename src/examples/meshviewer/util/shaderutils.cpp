/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#include "shaderutils.h"

 // Qt
#include <QtCore/QFile>
#include <QtCore/QRegularExpression>
#include <QtCore/QStringBuilder>
#include <QtCore/QTextStream>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Editor { namespace Util
{
	QString resolveShaderFile(QString full_path)
	{
		QRegularExpression dir_regex{ R"((.+/)(.+))" };
		QRegularExpressionMatch match;
		full_path.indexOf(dir_regex, 0, &match);
		VclCheck(match.hasMatch(), "Split is successfull.");

		QString dir = match.captured(1);
		QString path = match.captured(2);

		QFile shader_file{ dir + path };
		shader_file.open(QIODevice::ReadOnly | QIODevice::Text);
		VclCheck(shader_file.isOpen(), "Shader file is open.");

		// Resolve include files (only one level supported atm)
		QString builder;
		QTextStream textStream(&shader_file);

		QRegularExpression inc_regex{ R"(#.*include.*[<"](.+)[">])" };
		while (!textStream.atEnd())
		{
			auto curr_tok = textStream.readLine();

			QRegularExpressionMatch match_inc;
			if (curr_tok.indexOf(inc_regex, 0, &match_inc) >= 0 && match_inc.hasMatch())
			{
				QString included_file = resolveShaderFile(dir + match_inc.captured(1));
				builder = builder % included_file % "\n";
			}
			else if (curr_tok.indexOf("GL_GOOGLE_include_directive") >= 0)
			{
				continue;
			}
			else
			{
				builder = builder % curr_tok % "\n";
			}
		}

		shader_file.close();

		return builder;
	}

	Vcl::Graphics::Runtime::OpenGL::Shader createShader(Vcl::Graphics::Runtime::ShaderType type, QString path)
	{
		QString data = resolveShaderFile(path);

		return{ type, 0, data.toUtf8().data() };
	}
}}}
