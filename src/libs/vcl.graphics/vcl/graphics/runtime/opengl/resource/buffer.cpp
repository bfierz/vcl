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
#include <vcl/graphics/runtime/opengl/resource/buffer.h>

// C++ standard library

// VCL
#include <vcl/core/contract.h>

#ifdef VCL_OPENGL_SUPPORT

#define VCL_GL_BUFFER_COHERENCY

#if defined VCL_GL_BUFFER_COHERENCY
#	define VCL_GL_BUFFER_COHERENT_ACCESS GL_MAP_COHERENT_BIT
#else
#	define VCL_GL_BUFFER_COHERENT_ACCESS GL_NONE
#endif // VCL_GL_BUFFER_COHERENCY


namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Buffer::Buffer(const BufferDescription& desc, const BufferInitData* init_data)
	: Runtime::Buffer(desc)
	, Resource()
	{
		Require(implies(usage() == Usage::Immutable, cpuAccess().isAnySet() == false), "No CPU access requested for immutable buffer.");
		Require(implies(usage() == Usage::Dynamic, cpuAccess().isSet(CPUAccess::Read) == false), "Dynamic buffer is not mapped for reading.");
		Require(implies(init_data, init_data->SizeInBytes == desc.SizeInBytes), "Initialization data has same size as buffer.");

		GLenum flags = GL_NONE;
		GLenum map_flags = GL_NONE;
		switch (usage())
		{
		case Usage::Default:
			// Allows to use glBufferSubData
			flags |= GL_DYNAMIC_STORAGE_BIT;
			map_flags |= GL_NONE;
			break;

		case Usage::Immutable:
			flags |= GL_NONE;
			map_flags |= GL_NONE;
			break;

		case Usage::Dynamic:
			flags |= GL_MAP_PERSISTENT_BIT | VCL_GL_BUFFER_COHERENT_ACCESS;
			map_flags |= GL_MAP_PERSISTENT_BIT | VCL_GL_BUFFER_COHERENT_ACCESS;

			if (cpuAccess().isSet(CPUAccess::Write))
			{
				flags |= GL_MAP_WRITE_BIT;
				map_flags |= GL_MAP_WRITE_BIT;
			}

			break;

		case Usage::Staging:
			flags |= GL_DYNAMIC_STORAGE_BIT | GL_MAP_PERSISTENT_BIT | VCL_GL_BUFFER_COHERENT_ACCESS;
			map_flags |= GL_MAP_PERSISTENT_BIT | VCL_GL_BUFFER_COHERENT_ACCESS;

			if (cpuAccess().isSet(CPUAccess::Write))
			{
				flags |= GL_MAP_WRITE_BIT;
				map_flags |= GL_MAP_WRITE_BIT;
			}

			if (cpuAccess().isSet(CPUAccess::Read))
			{
				flags |= GL_MAP_READ_BIT;
				map_flags |= GL_MAP_READ_BIT;
			}

			break;
		}

		// Allocate a GL buffer ID
		glCreateBuffers(1, &_glId);

		// Allocate GPU memory
		_sizeInBytes = desc.SizeInBytes;
		void* init_data_ptr = init_data ? init_data->Data : nullptr;
		glNamedBufferStorage(_glId, desc.SizeInBytes, init_data_ptr, flags);

		// Persistently map the buffer to the host memory, if usage allows it
		if (usage() == Usage::Dynamic && cpuAccess().isSet(CPUAccess::Write))
		{
#if !defined VCL_GL_BUFFER_COHERENCY
			if (cpuAccess().isSet(CPUAccess::Write))
				map_flags |= GL_MAP_FLUSH_EXPLICIT_BIT;
#endif // VCL_GL_BUFFER_COHERENCY

			_mappedPtr = glMapNamedBufferRange(_glId, 0, _sizeInBytes, map_flags);
		}
		else if (usage() == Usage::Staging && cpuAccess().isAnySet())
		{
#if !defined VCL_GL_BUFFER_COHERENCY
			if (cpuAccess().isSet(CPUAccess::Write))
				map_flags |= GL_MAP_FLUSH_EXPLICIT_BIT;
#endif // VCL_GL_BUFFER_COHERENCY

			_mappedPtr = glMapNamedBufferRange(_glId, 0, _sizeInBytes, map_flags);
		}

		Ensure(_glId > 0, "GL buffer is created.");
		Ensure(implies(usage() == Usage::Dynamic, _mappedPtr), "Buffer is mapped.");
	}

	Buffer::~Buffer()
	{
		Require(_glId > 0, "GL buffer is created.");
		Require(implies(usage() == Usage::Dynamic, _mappedPtr), "GL memory is mapped.");

		if (_mappedPtr)
		{
			glUnmapNamedBuffer(_glId);

			// Reset the mapped pointer
			_mappedPtr = nullptr;
		}

		if (_glId > 0)
		{
			glDeleteBuffers(1, &_glId);

			// Reset the buffer id
			_glId = 0;
		}

		Ensure(!_mappedPtr, "Buffer is not mapped.");
		Ensure(_glId == 0, "GL buffer is cleaned up.");
	}

	void* Buffer::map(size_t offset, size_t length, MapOptions access)
	{
		Require(implies(usage() == Usage::Dynamic || usage() == Usage::Staging, _mappedPtr), "GL memory is mapped.");
		Require(!_mapped, "Buffer is not mapped.");

		_mapped = true;

		Ensure(implies(usage() == Usage::Dynamic || usage() == Usage::Staging, _mapped), "Buffer is mapped.");

		return _mappedPtr;
	}

	void Buffer::unmap()
	{
		Require(implies(usage() == Usage::Dynamic || usage() == Usage::Staging, _mappedPtr), "GL memory is mapped.");
		Require(implies(usage() == Usage::Dynamic || usage() == Usage::Staging, _mapped), "Buffer is mapped.");

#if !defined VCL_GL_BUFFER_COHERENCY
		// If access is not using the coherency flag, we have to 
		// make sure that we are still providing coherent memory access
		if (cpuAccess().isSet(CPUAccess::Write))
			glFlushMappedNamedBufferRange(_glId, 0, _sizeInBytes);
#endif // VCL_GL_BUFFER_COHERENCY

		// Mark buffer as unmapped
		_mapped = false;

		Ensure(!_mapped, "Buffer is not mapped.");
	}
}}}}
#endif // VCL_OPENGL_SUPPORT
