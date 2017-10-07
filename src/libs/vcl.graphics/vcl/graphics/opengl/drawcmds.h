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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace OpenGL
{
	struct DrawCommand
	{
		DrawCommand(int count, int inst_count, int first, int base_inst)
		: Count(static_cast<uint32_t>(count))
		, InstanceCount(static_cast<uint32_t>(inst_count))
		, First(static_cast<uint32_t>(first))
		, BaseInstance(static_cast<uint32_t>(base_inst))
		{}

		uint32_t Count;
		uint32_t InstanceCount;
		uint32_t First;
		uint32_t BaseInstance;
	};

	struct DrawIndexedCommand
	{
		DrawIndexedCommand(int count, int inst_count, int first_index, int base_vertex, int base_inst)
		: Count(static_cast<uint32_t>(count))
		, InstanceCount(static_cast<uint32_t>(inst_count))
		, FirstIndex(static_cast<uint32_t>(first_index))
		, BaseVertex(static_cast<uint32_t>(base_vertex))
		, BaseInstance(static_cast<uint32_t>(base_inst))
		{}

		uint32_t Count;
		uint32_t InstanceCount;
		uint32_t FirstIndex;
		uint32_t BaseVertex;
		uint32_t BaseInstance;
	};

}}}

#endif // VCL_OPENGL_SUPPORT
