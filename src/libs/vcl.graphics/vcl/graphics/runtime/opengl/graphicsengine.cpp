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
#include <vcl/graphics/runtime/opengl/graphicsengine.h>

// VCL
#include <vcl/util/hashedstring.h>
#include <vcl/graphics/opengl/drawcmds.h>
#include <vcl/graphics/opengl/gl.h>
#include <vcl/graphics/runtime/opengl/resource/texture.h>
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/graphics/runtime/opengl/state/framebuffer.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/graphics/runtime/opengl/state/sampler.h>
#include <vcl/math/ceil.h>

namespace
{
	GLenum toGLenum(Vcl::Graphics::Runtime::PrimitiveType topology)
	{
		using Vcl::Graphics::Runtime::PrimitiveType;

		switch (topology)
		{
		case PrimitiveType::Undefined:
			return GL_INVALID_ENUM;
		case PrimitiveType::Pointlist:
			return GL_POINTS;
		case PrimitiveType::Linelist:
			return GL_LINES;
		case PrimitiveType::Linestrip:
			return GL_LINE_STRIP;
		case PrimitiveType::Trianglelist:
			return GL_TRIANGLES;
		case PrimitiveType::Trianglestrip:
			return GL_TRIANGLE_STRIP;
		case PrimitiveType::LinelistAdj:
			return GL_LINES_ADJACENCY;
		case PrimitiveType::LinestripAdj:
			return GL_LINE_STRIP_ADJACENCY;
		case PrimitiveType::TrianglelistAdj:
			return GL_TRIANGLES_ADJACENCY;
		case PrimitiveType::TrianglestripAdj:
			return GL_TRIANGLE_STRIP_ADJACENCY;
		case PrimitiveType::Patch:
			return GL_PATCHES;
		default: { VclDebugError("Enumeration value is valid."); return GL_INVALID_ENUM; }
		}
	}	

}

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	Fence::Fence(GLsync sync)
	: _sync(sync)
	{

	}

	Fence::Fence(Fence&& rhs)
	{
		std::swap(_sync, rhs._sync);
	}

	Fence::~Fence()
	{
		glDeleteSync(_sync);
		_sync = nullptr;
	}

	Fence& Fence::operator=(Fence&& rhs)
	{
		std::swap(_sync, rhs._sync);
		return *this;
	}

	StagingArea::StagingArea(Frame* frame)
	: _frame(frame)
	{
		VclRequire(frame, "Owner is set.");

		BufferDescription desc;
		desc.Usage = ResourceUsage::Staging;
		desc.CPUAccess = ResourceAccess::Read;
		desc.SizeInBytes = 1000 * 4096;

		_stagingBuffer = make_owner<Buffer>(desc);
		_hostBuffer = std::make_unique<char[]>(desc.SizeInBytes);
	}

	void StagingArea::transfer()
	{
		using Vcl::Mathematics::ceil;

		if (_requests.empty())
			return;

		// Compute the transfer size
		size_t size = ceil(_requests.back().offset() + _requests.back().size(), _alignment);

		// Do the transfer
		_stagingBuffer->copyTo(_hostBuffer.get(), 0, 0, size);

		// Clear the requests
		_stagingBufferOffset = 0;
		_requests.clear();
	}

	BufferView StagingArea::copyFrom(const BufferView& view)
	{
		using Vcl::Mathematics::ceil;

		// Request size from the staging buffer
		size_t size_incr = ceil(view.size(), _alignment);
		
		// Check the available size
		updateIfNeeded(size_incr);
		
		// Copy the content to the readback buffer
		dynamic_cast<const OpenGL::Buffer&>(view.owner()).copyTo(*_stagingBuffer, 0, _stagingBufferOffset, view.size());

		// View on the copied data
		BufferView staged_area{ _stagingBuffer, _stagingBufferOffset, view.size(), _hostBuffer.get() };
		_stagingBufferOffset += size_incr;

		// Store the request
		_requests.emplace_back(staged_area);

		return staged_area;
	}

	BufferView StagingArea::copyFrom(const TextureView& view)
	{
		using Vcl::Mathematics::ceil;

		// Request size from the staging buffer
		size_t size_incr = ceil(view.sizeInBytes(), _alignment);

		// Check the available size
		updateIfNeeded(size_incr);

		// Copy the content to the readback buffer
		dynamic_cast<const OpenGL::Texture&>(view).copyTo(*_stagingBuffer, _stagingBufferOffset);

		// View on the copied data
		BufferView staged_area{ _stagingBuffer, _stagingBufferOffset, view.sizeInBytes(), _hostBuffer.get() };
		_stagingBufferOffset += size_incr;

		// Store the request
		_requests.emplace_back(staged_area);

		return staged_area;
	}

	void StagingArea::updateIfNeeded(size_t size)
	{
		if (_stagingBuffer->sizeInBytes() < _stagingBufferOffset + size)
		{
			BufferDescription desc;
			desc.Usage = ResourceUsage::Staging;
			desc.CPUAccess = ResourceAccess::Read;
			desc.SizeInBytes = (uint32_t) _stagingBuffer->sizeInBytes() * 2;

			// Allocate a new buffer
			auto tmp = make_owner<Buffer>(desc);
			_stagingBuffer->copyTo(*tmp);

			// Mark the buffer for deletion
			_frame->queueBufferForDeletion(std::move(_stagingBuffer));

			_stagingBuffer = std::move(tmp);
			_hostBuffer = std::make_unique<char[]>(desc.SizeInBytes);
		}
	}

	Frame::Frame()
	: _readBackStage(this)
	{
		// Allocate a junk of 512 KB for constant buffers per frame
		BufferDescription cbuffer_desc;
		cbuffer_desc.SizeInBytes = 1 << 19;
		cbuffer_desc.CPUAccess = ResourceAccess::Write;
		cbuffer_desc.Usage = ResourceUsage::Dynamic;

		_constantBuffer = make_owner<Buffer>(cbuffer_desc, true, true);

		// Allocate a junk of 16 MB for linear memory per frame
		BufferDescription linbuffer_desc;
		linbuffer_desc.SizeInBytes = 1 << 24;
		linbuffer_desc.CPUAccess = ResourceAccess::Write;
		linbuffer_desc.Usage = ResourceUsage::Dynamic;

		_linearMemoryBuffer = make_owner<Buffer>(linbuffer_desc, true, true);
	}

	void Frame::mapBuffers()
	{
		_mappedConstantBuffer = _constantBuffer->map(0, _constantBuffer->sizeInBytes(), ResourceAccess::Write, MapOptions::Persistent | MapOptions::CoherentWrite | MapOptions::Unsynchronized | MapOptions::InvalidateBuffer);
		_mappedLinearMemory = _linearMemoryBuffer->map(0, _linearMemoryBuffer->sizeInBytes(), ResourceAccess::Write, MapOptions::Persistent | MapOptions::CoherentWrite | MapOptions::Unsynchronized | MapOptions::InvalidateBuffer);
	}

	void Frame::unmapBuffers()
	{
		_linearMemoryBuffer->unmap();
		_constantBuffer->unmap();
	}
	
	void Frame::setRenderTargets(std::span<Runtime::Texture*> colour_targets, Runtime::Texture* depth_target)
	{
		// Calculate the hash for the set of textures
		std::array<void*, 9> ptrs;
		ptrs[0] = depth_target;
		for (size_t i = 0; i < colour_targets.size(); i++)
		{
			ptrs[i + 1] = colour_targets[i];
		}

		const unsigned int hash = Vcl::Util::calculateFnv1a32(reinterpret_cast<char*>(ptrs.data()), ptrs.end() - ptrs.begin());

		// Find the FBO in the cache
		auto cache_entry = _fbos.find(hash);
		if (cache_entry != _fbos.end())
		{
			cache_entry->second.bind();
		}
		else
		{
			const Runtime::Texture* colours[8];
			for (size_t i = 0; i < colour_targets.size(); i++)
			{
				colours[i] = colour_targets[i];
			}
			auto depth = depth_target;
			auto new_entry = _fbos.emplace(hash, OpenGL::Framebuffer{ colours, (size_t) colour_targets.size(), depth });
			_currentFramebuffer = &new_entry.first->second;
			_currentFramebuffer->bind();
		}
	}

	void Frame::setRenderTargets(std::span<ref_ptr<Runtime::Texture>> colour_targets, ref_ptr<Runtime::Texture> depth_target)
	{
		if ((colour_targets.size() == 0 || !colour_targets[0]) && !depth_target)
		{
			_currentFramebuffer = nullptr;
			return;
		}

		// Calculate the hash for the set of textures
		std::array<void*, 9> ptrs;
		ptrs[0] = depth_target.get();
		for (size_t i = 0; i < colour_targets.size(); i++)
		{
			ptrs[i + 1] = colour_targets[i].get();
		}

		const unsigned int hash = Vcl::Util::calculateFnv1a32(reinterpret_cast<char*>(ptrs.data()), ptrs.end() - ptrs.begin());

		// Find the FBO in the cache
		auto cache_entry = _fbos.find(hash);
		if (cache_entry != _fbos.end())
		{
			cache_entry->second.bind();
		}
		else
		{
			const Runtime::Texture* colours[8];
			for (size_t i = 0; i < colour_targets.size(); i++)
			{
				colours[i] = colour_targets[i].get();
			}
			auto depth = depth_target.get();
			auto new_entry = _fbos.emplace(hash, OpenGL::Framebuffer{ colours, (size_t) colour_targets.size(), depth });
			_currentFramebuffer = &new_entry.first->second;
			_currentFramebuffer->bind();
		}
	}

	void Frame::queueBufferForDeletion(owner_ptr<Buffer> buffer)
	{
		_bufferRecycleQueue.push_back(std::move(buffer));
	}

	GraphicsEngine::GraphicsEngine()
	{
		using Vcl::Graphics::OpenGL::GL;

		// Read the device dependent constant
		_cbufferAlignment = GL::getInteger(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT);

		// Create three frames:
		// 1. where the CPU writes to
		// 2. which the GPU driver is processing
		// 3. which the GPU is processing
		_frames.resize(_numConcurrentFrames);
	}
	
	owner_ptr<Runtime::Texture> GraphicsEngine::createResource(const Texture2DDescription& desc)
	{
		return make_owner<OpenGL::Texture2D>(desc);
	}
	
	owner_ptr<Runtime::Buffer> GraphicsEngine::createResource(const BufferDescription& desc)
	{
		return make_owner<OpenGL::Buffer>(desc);
	}

	void GraphicsEngine::beginFrame()
	{
		// Fetch the frame we want to use for the current rendering frame
		_currentFrame = &_frames[currentFrame() % _numConcurrentFrames];

		// Wait until the requested frame is done with processing
		if (_currentFrame->fence()->isValid())
		{
			auto wait_result = glClientWaitSync(*_currentFrame->fence(), GL_NONE, 1000000000);
			if (wait_result != GL_CONDITION_SATISFIED || wait_result != GL_ALREADY_SIGNALED)
			{
				if (wait_result == GL_TIMEOUT_EXPIRED)
					throw std::runtime_error("Waiting for sync object expired.");

				if (wait_result == GL_WAIT_FAILED)
					throw std::runtime_error("Waiting for sync object failed.");
			}
		}

		// Reset the staging area to use it again for the new frame
		_currentFrame->executeReadbackRequests();

		// Map the per frame constant buffer
		_currentFrame->mapBuffers();

		// Reset the begin pointer for buffer chunk requests
		_cbufferOffset = 0;
		_linearBufferOffset = 0;

		// Execute the stored commands
		{
			std::unique_lock<std::mutex> lck{ _cmdMutex };

			for (auto& cmd : _genericCmds)
			{
				cmd();
			}

			_genericCmds.clear();
		}
	}

	void GraphicsEngine::endFrame()
	{
		// Unmap the constant buffer
		_currentFrame->unmapBuffers();

		// Queue a fence sync object to mark the end of the frame
		_currentFrame->setFence(glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, GL_NONE));

		// Clear the current frame to mark that not frame is active anymore
		_currentFrame = nullptr;

		// Flush the OpenGL command pipeline to ensure command processing before the
		// UI command list is build
		glFlush();
		
		// Increment the frame counter to indicate the start of the next frame
		incrFrameCounter();
	}

	BufferView GraphicsEngine::requestPerFrameConstantBuffer(size_t size)
	{
		BufferView view{ _currentFrame->constantBuffer(), _cbufferOffset, size, _currentFrame->mappedConstantBuffer() };

		// Calculate the next offset		
		size_t aligned_size =  ((size + (_cbufferAlignment - 1)) / _cbufferAlignment) * _cbufferAlignment;
		_cbufferOffset += aligned_size;

		return view;
	}

	BufferView GraphicsEngine::requestPerFrameLinearMemory(size_t size)
	{
		BufferView view{ _currentFrame->linearMemoryBuffer(), _linearBufferOffset, size, _currentFrame->mappedLinearMemoryBuffer() };

		// Calculate the next offset		
		size_t aligned_size = ((size + (_cbufferAlignment - 1)) / _cbufferAlignment) * _cbufferAlignment;
		_linearBufferOffset += aligned_size;

		return view;
	}

	void GraphicsEngine::enqueueReadback(const Runtime::Texture& tex, std::function<void(const BufferView&)> callback)
	{
		_currentFrame->enqueueReadback(tex, std::move(callback));
	}

	void GraphicsEngine::enqueueCommand(std::function<void(void)> cmd)
	{
		std::unique_lock<std::mutex> lck{ _cmdMutex };

		_genericCmds.emplace_back(std::move(cmd));
	}

	void GraphicsEngine::setRenderTargets(std::span<const ref_ptr<Runtime::Texture>> colour_targets, ref_ptr<Runtime::Texture> depth_target)
	{
		// Set the render targets for the current frame
		ref_ptr<Runtime::Texture> colours[8];
		for (size_t i = 0; i < colour_targets.size(); i++)
		{
			colours[i] = colour_targets[i];
		}

		_currentFrame->setRenderTargets(std::make_span(colours, colour_targets.size()), depth_target);
	}

	void GraphicsEngine::setConstantBuffer(int idx, BufferView view)
	{
		auto& buffer = static_cast<const OpenGL::Buffer&>(view.owner());
		glBindBufferRange(GL_UNIFORM_BUFFER, idx, buffer.id(), view.offset(), view.size());
	}
	
	void GraphicsEngine::setVertexBuffer(int idx, const Runtime::Buffer& buffer, int offset, int stride)
	{
		auto& gl_buffer = static_cast<const OpenGL::Buffer&>(buffer);
		glBindVertexBuffer(idx, gl_buffer.id(), offset, stride);
	}

	void GraphicsEngine::setSampler(int idx, const Runtime::Sampler& sampler)
	{
		auto& gl_sampler = static_cast<const OpenGL::Sampler&>(sampler);

		GLuint sampler_id = gl_sampler.id();
		glBindSampler(idx, sampler_id);
	}

	void GraphicsEngine::setSamplers(int idx, std::span<const ref_ptr<Runtime::Sampler>> samplers)
	{
	}

	void GraphicsEngine::setTexture(int idx, const Runtime::Texture& texture)
	{
		auto& gl_texture = static_cast<const OpenGL::Texture&>(texture);
	
		GLuint tex_id = gl_texture.id();
		glBindTextures(idx, 1, &tex_id);
	}

	void GraphicsEngine::setTextures(int idx, std::span<const ref_ptr<Runtime::Texture>> textures)
	{
	}

	void GraphicsEngine::setPipelineState(ref_ptr<Runtime::PipelineState> state)
	{
		auto gl_state = static_pointer_cast<OpenGL::PipelineState>(state);

		gl_state->bind();
	}
	
	void GraphicsEngine::pushConstants(void* data, size_t size)
	{
		VclRequire(size <= 128, "Push-constants less than 128 bytes");

		auto buffer = requestPerFrameConstantBuffer(size);
		memcpy(buffer.data(), data, size);

		setConstantBuffer(_pushConstantBufferIndex, buffer);
	}
	
	void GraphicsEngine::clear(int idx, const Eigen::Vector4f& colour)
	{
		if (_currentFrame->currentFramebuffer())
		{
			_currentFrame->currentFramebuffer()->clear(idx, colour);
		}
		else
		{
			glClearBufferfv(GL_COLOR, idx, colour.data());
		}
	}
	void GraphicsEngine::clear(int idx, const Eigen::Vector4i& colour)
	{
		if (_currentFrame->currentFramebuffer())
		{
			_currentFrame->currentFramebuffer()->clear(idx, colour);
		}
		else
		{
			glClearBufferiv(GL_COLOR, idx, colour.data());
		}
	}
	void GraphicsEngine::clear(int idx, const Eigen::Vector4ui& colour)
	{
		if (_currentFrame->currentFramebuffer())
		{
			_currentFrame->currentFramebuffer()->clear(idx, colour);
		}
		else
		{
			glClearBufferuiv(GL_COLOR, idx, colour.data());
		}
	}
	void GraphicsEngine::clear(float depth, int stencil)
	{
		if (_currentFrame->currentFramebuffer())
		{
			_currentFrame->currentFramebuffer()->clear(depth, stencil);
		}
		else
		{
			glClearBufferfi(GL_DEPTH_STENCIL, 0, depth, stencil);
		}
	}
	void GraphicsEngine::clear(float depth)
	{
		if (_currentFrame->currentFramebuffer())
		{
			_currentFrame->currentFramebuffer()->clear(depth);
		}
		else
		{
			glClearBufferfv(GL_DEPTH, 0, &depth);
		}
	}
	void GraphicsEngine::clear(int stencil)
	{
		if (_currentFrame->currentFramebuffer())
		{
			_currentFrame->currentFramebuffer()->clear(stencil);
		}
		else
		{
			glClearBufferiv(GL_STENCIL, 0, &stencil);
		}
	}
	
	void GraphicsEngine::setPrimitiveType(PrimitiveType type, int nr_vertices)
	{	
		_currentPrimitiveType = type;
		_currentNrPatchVertices = nr_vertices;

		if (_currentPrimitiveType == PrimitiveType::Patch)
		{
			glPatchParameteri(GL_PATCH_VERTICES, _currentNrPatchVertices);
		}
	}
	
	void GraphicsEngine::draw(int count, int first, int instance_count, int base_instance)
	{
		GLenum mode = toGLenum(_currentPrimitiveType);
		//Graphics::OpenGL::DrawCommand cmd{ count, instance_count, first, base_instance };
		//glDrawArraysIndirect(mode, &cmd);
		glDrawArraysInstancedBaseInstance(mode, first, count, instance_count, base_instance);
	}

	void GraphicsEngine::drawIndexed(int count, int first_index, int instance_count, int base_vertex, int base_instance)
	{
		GLenum mode = toGLenum(_currentPrimitiveType);
		//Graphics::OpenGL::DrawIndexedCommand cmd{ count, first_index, instance_count, base_vertex, base_instance };
		//glDrawElementsIndirect(mode, GL_UNSIGNED_INT, &cmd);
		glDrawElementsInstancedBaseInstance(mode, count, GL_UNSIGNED_INT, nullptr, instance_count, base_instance);
	}

}}}}
