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

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	BufferBindPoint::BufferBindPoint(GLenum target, GLuint id)
	: _target(target)
	, _id(id)
	{
		glBindBuffer(_target, _id);
	}
	BufferBindPoint::~BufferBindPoint()
	{
		glBindBuffer(_target, GL_NONE);
	}

	Buffer::Buffer(const BufferDescription& desc, bool allowPersistentMapping, bool allowCoherentMapping, const BufferInitData* init_data)
	: Runtime::Buffer(desc.SizeInBytes, desc.Usage, desc.CPUAccess)
	, Resource()
	, _allowPersistentMapping(allowPersistentMapping)
	, _allowCoherentMapping(allowCoherentMapping)
	{
		Require(implies(usage() == Usage::Immutable, cpuAccess().isAnySet() == false), "No CPU access requested for immutable buffer.");
		Require(implies(usage() == Usage::Dynamic, cpuAccess().isSet(CPUAccess::Read) == false), "Dynamic buffer is not mapped for reading.");
		Require(implies(init_data, init_data->SizeInBytes == desc.SizeInBytes), "Initialization data has same size as buffer.");
		Require(implies(_allowCoherentMapping, _allowPersistentMapping), "A coherent buffer access is persistent.");

		GLenum flags = GL_NONE;
		switch (usage())
		{
		case Usage::Default:
			// Allows to use glBufferSubData
			flags |= GL_DYNAMIC_STORAGE_BIT;
			break;

		case Usage::Immutable:
			flags |= GL_NONE;
			break;

		case Usage::Dynamic:
			// Dynamic buffers are always map writable
			flags |= GL_MAP_WRITE_BIT;

			if (_allowPersistentMapping)
				flags |= GL_MAP_PERSISTENT_BIT;

			if (_allowCoherentMapping)
				flags |= GL_MAP_COHERENT_BIT;

			break;

		case Usage::Staging:
			flags |= GL_DYNAMIC_STORAGE_BIT;

			if (_allowPersistentMapping)
				flags |= GL_MAP_PERSISTENT_BIT;

			if (_allowCoherentMapping)
				flags |= GL_MAP_COHERENT_BIT;

			if (cpuAccess().isSet(CPUAccess::Write))
				flags |= GL_MAP_WRITE_BIT;

			if (cpuAccess().isSet(CPUAccess::Read))
				flags |= GL_MAP_READ_BIT;

			break;
		}
		
#if defined(VCL_GL_ARB_direct_state_access)
		// Allocate a GL buffer ID
		glCreateBuffers(1, &_glId);

		// Allocate GPU memory
		void* init_data_ptr = init_data ? init_data->Data : nullptr;
		glNamedBufferStorage(_glId, desc.SizeInBytes, init_data_ptr, flags);
#elif defined(VCL_GL_EXT_direct_state_access)
		// Allocate a GL buffer ID
		glGenBuffers(1, &_glId);

		// Allocate GPU memory
		void* init_data_ptr = init_data ? init_data->Data : nullptr;
		glNamedBufferStorageEXT(_glId, desc.SizeInBytes, init_data_ptr, flags);
#endif
		
		Ensure(_glId > 0, "GL buffer is created.");
	}

	Buffer::~Buffer()
	{
		Require(_glId > 0, "GL buffer is created.");

		if (_mappedAccess.isAnySet())
		{
#if defined(VCL_GL_ARB_direct_state_access)
			glUnmapNamedBuffer(_glId);
#elif defined(VCL_GL_EXT_direct_state_access)
			glUnmapNamedBufferEXT(_glId);
#endif

			// Reset the mapped indicator
			_mappedAccess.clear();
			_mappedOptions.clear();
			_mappedOffset = 0;
			_mappedSize = 0;
		}

		if (_glId > 0)
		{
			glDeleteBuffers(1, &_glId);

			// Reset the buffer id
			_glId = 0;
		}

		Ensure(_glId == 0, "GL buffer is cleaned up.");
	}

	BufferBindPoint Buffer::bind(GLenum target)
	{
		return{ target, _glId };
	}

	void* Buffer::map(size_t offset, size_t length, Flags<CPUAccess> access, Flags<MapOptions> options)
	{
		Require(_glId > 0, "GL buffer is created.");
		Require(!_mappedAccess.isAnySet(), "Buffer is not mapped");
		Require(usage() == Usage::Dynamic || usage() == Usage::Staging, "GL memory is mappable");
		Require(offset + length <= sizeInBytes(), "Map request lies in range");

		Require(implies(options.isSet(MapOptions::CoherentWrite), _allowCoherentMapping && options.isSet(MapOptions::Persistent)), "Coherent access was configured upon creation");
		Require(implies(options.isSet(MapOptions::Persistent), _allowPersistentMapping), "Persistent mapping was configured upon creation");
		Require(implies(options.isSet(MapOptions::ExplicitFlush), access.isSet(CPUAccess::Write)), "Explicit flush control only valid when mapped write");
		Require(implies(options.isSet(MapOptions::InvalidateBuffer), !access.isSet(CPUAccess::Read)), "Invalidation is valid on write");
		Require(implies(options.isSet(MapOptions::InvalidateRange), !access.isSet(CPUAccess::Read)), "Invalidation is valid on write");

		Require(implies(access.isSet(CPUAccess::Read), cpuAccess().isSet(CPUAccess::Read)), "Read access was requested at initialization");
		Require(implies(access.isSet(CPUAccess::Write), cpuAccess().isSet(CPUAccess::Write)), "Write access was requested at initialization");

		// Mapped pointer
		void* mappedPtr = nullptr;

		// Map flags
		GLenum map_flags = GL_NONE;
		if (options.isSet(MapOptions::InvalidateBuffer))
			map_flags |= GL_MAP_INVALIDATE_BUFFER_BIT;
		else if (options.isSet(MapOptions::InvalidateRange))
			map_flags |= GL_MAP_INVALIDATE_RANGE_BIT;

		// Persistently map the buffer to the host memory, if usage allows it
		if (usage() == Usage::Dynamic && access.isSet(CPUAccess::Write))
		{
			if (options.isSet(MapOptions::Persistent))
				map_flags |= GL_MAP_PERSISTENT_BIT;

			if (options.isSet(MapOptions::CoherentWrite))
				map_flags |= GL_MAP_COHERENT_BIT;
			else if (options.isSet(MapOptions::ExplicitFlush) && access.isSet(CPUAccess::Write))
				map_flags |= GL_MAP_FLUSH_EXPLICIT_BIT;

			if (access.isSet(CPUAccess::Write))
				map_flags |= GL_MAP_WRITE_BIT;

#if defined(VCL_GL_ARB_direct_state_access)
			mappedPtr = glMapNamedBufferRange(_glId, offset, length, map_flags);
#elif defined(VCL_GL_EXT_direct_state_access)
			mappedPtr = glMapNamedBufferRangeEXT(_glId, offset, length, map_flags);
#endif

			_mappedAccess = access;
			_mappedOptions = options;
			_mappedOffset = offset;
			_mappedSize = length;
		}
		else if (usage() == Usage::Staging && cpuAccess().isAnySet())
		{
			if (options.isSet(MapOptions::Persistent))
				map_flags |= GL_MAP_PERSISTENT_BIT;

			if (options.isSet(MapOptions::CoherentWrite))
				map_flags |= GL_MAP_COHERENT_BIT;
			else if (options.isSet(MapOptions::ExplicitFlush) && access.isSet(CPUAccess::Write))
				map_flags |= GL_MAP_FLUSH_EXPLICIT_BIT;

			if (access.isSet(CPUAccess::Write))
				map_flags |= GL_MAP_WRITE_BIT;

			if (access.isSet(CPUAccess::Read))
				map_flags |= GL_MAP_READ_BIT;

#if defined(VCL_GL_ARB_direct_state_access)
			mappedPtr = glMapNamedBufferRange(_glId, offset, length, map_flags);
#elif defined(VCL_GL_EXT_direct_state_access)
			mappedPtr = glMapNamedBufferRangeEXT(_glId, offset, length, map_flags);
#endif

			_mappedAccess = access;
			_mappedOptions = options;
			_mappedOffset = offset;
			_mappedSize = length;
		}
		
		Ensure(implies(usage() == Usage::Dynamic || usage() == Usage::Staging, _mappedAccess.isAnySet()), "Buffer is mapped.");
		AssertBlock
		{
			GLint64 min_align = 0;
			glGetInteger64v(GL_MIN_MAP_BUFFER_ALIGNMENT, &min_align);
			EnsureEx(((ptrdiff_t) mappedPtr - offset) % min_align == 0, "Mapped pointers are aligned correctly.", "Offset: {}, Minimum aligment: {}", offset, min_align);
		}

		return mappedPtr;
	}

	void Buffer::flushRange(size_t offset, size_t length)
	{
		Require(_glId > 0, "GL buffer is created.");
		Require(_mappedAccess.isSet(CPUAccess::Write) && _mappedOptions.isSet(MapOptions::ExplicitFlush), "Buffer is mapped with explicit flush");
		Require(offset + length <= _mappedSize, "Flush request lies in mapped range");

		// If access is not using the coherency flag, we have to 
		// make sure that we are still providing coherent memory access
#if defined(VCL_GL_ARB_direct_state_access)
		glFlushMappedNamedBufferRange(_glId, offset, length);
#elif defined(VCL_GL_EXT_direct_state_access)
		glFlushMappedNamedBufferRangeEXT(_glId, offset, length);
#endif

	}

	void Buffer::unmap()
	{
		Require(_glId > 0, "GL buffer is created.");
		Require(implies(usage() == Usage::Dynamic || usage() == Usage::Staging, _mappedAccess.isAnySet()), "Buffer is mapped.");

		// Mark buffer as unmapped
		_mappedAccess.clear();
		_mappedOptions.clear();
		_mappedOffset = 0;
		_mappedSize = 0;

		// Unmap the buffer
#if defined(VCL_GL_ARB_direct_state_access)
		GLboolean success = glUnmapNamedBuffer(_glId);
#elif defined(VCL_GL_EXT_direct_state_access)
		GLboolean success = glUnmapNamedBufferEXT(_glId);
#endif
		if (!success)
		{
			// Memory was lost
			throw gl_memory_error{ "Video memory was trashed" };
		}
		
		Ensure(!_mappedAccess.isAnySet(), "Buffer is not mapped.");
	}

	void Buffer::copyTo(void* dst, size_t srcOffset, size_t dstOffset, size_t size) const
	{
		Require(_glId > 0, "GL buffer is created.");
		Require(implies(size < std::numeric_limits<size_t>::max(), srcOffset + size <= sizeInBytes()), "Size to copy is valid");

		if (size == std::numeric_limits<size_t>::max())
			size = sizeInBytes() - srcOffset;

		glGetNamedBufferSubData(_glId, srcOffset, size, (char*) dst + dstOffset);
	}

	void Buffer::copyTo(Buffer& target, size_t srcOffset, size_t dstOffset, size_t size) const
	{
		Require(_glId > 0, "GL buffer is created.");
		Require(target.id() > 0, "GL buffer is created.");
		Require(sizeInBytes() <= target.sizeInBytes(), "Size to copy is valid");
		Require(implies(size < std::numeric_limits<size_t>::max(), srcOffset + size <= sizeInBytes()), "Size to copy is valid");
		Require(implies(size < std::numeric_limits<size_t>::max(), dstOffset + size <= target.sizeInBytes()), "Size to copy is valid");

		if (size == std::numeric_limits<size_t>::max())
			size = sizeInBytes() - srcOffset;
		Check(dstOffset + size <= target.sizeInBytes(), "Size to copy is valid");
		
#if defined(VCL_GL_ARB_direct_state_access)
		glCopyNamedBufferSubData(_glId, target.id(), srcOffset, dstOffset, size);
#elif defined(VCL_GL_EXT_direct_state_access)
		glNamedCopyBufferSubDataEXT(_glId, target.id(), srcOffset, dstOffset, size);
#endif

	}
}}}}
#endif // VCL_OPENGL_SUPPORT
