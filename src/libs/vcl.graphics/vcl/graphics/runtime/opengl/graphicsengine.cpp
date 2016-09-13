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

// C++ standard library
#include <iostream>

// VCL
#include <vcl/graphics/opengl/gl.h>
#include <vcl/graphics/runtime/opengl/resource/texture.h>
#include <vcl/graphics/runtime/opengl/state/framebuffer.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/math/ceil.h>

namespace
{
	void VCL_CALLBACK OpenGLDebugMessageCallback
	(
		GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const GLchar* message,
		const void* user_param
	)
	{
		VCL_UNREFERENCED_PARAMETER(length);
		VCL_UNREFERENCED_PARAMETER(user_param);

		std::cout << "Source: ";
		switch (source)
		{
		case GL_DEBUG_SOURCE_API:
			std::cout << "API";
			break;
		case GL_DEBUG_SOURCE_SHADER_COMPILER:
			std::cout << "Shader Compiler";
			break;
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
			std::cout << "Window System";
			break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:
			std::cout << "Third Party";
			break;
		case GL_DEBUG_SOURCE_APPLICATION:
			std::cout << "Application";
			break;
		case GL_DEBUG_SOURCE_OTHER:
			std::cout << "Other";
			break;
		}

		std::cout << ", Type: ";
		switch (type)
		{
		case GL_DEBUG_TYPE_ERROR:
			std::cout << "Error";
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			std::cout << "Deprecated Behavior";
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			std::cout << "Undefined Behavior";
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			std::cout << "Performance";
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			std::cout << "Portability";
			break;
		case GL_DEBUG_TYPE_OTHER:
			std::cout << "Other";
			break;
		case GL_DEBUG_TYPE_MARKER:
			std::cout << "Marker";
			break;
		case GL_DEBUG_TYPE_PUSH_GROUP:
			std::cout << "Push Group";
			break;
		case GL_DEBUG_TYPE_POP_GROUP:
			std::cout << "Pop Group";
			break;
		}

		std::cout << ", Severity: ";
		switch (severity)
		{
		case GL_DEBUG_SEVERITY_HIGH:
			std::cout << "High";
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			std::cout << "Medium";
			break;
		case GL_DEBUG_SEVERITY_LOW:
			std::cout << "Low";
			break;
		case GL_DEBUG_SEVERITY_NOTIFICATION:
			std::cout << "Notification";
			break;
		}

		std::cout << ", ID: " << id;
		std::cout << ", Message: " << message << std::endl;
	}

	unsigned int calculateFNV(gsl::span<char> str)
	{
		unsigned int hash = 2166136261u;

		for (ptrdiff_t i = 0; i < str.length(); ++i)
		{
			hash ^= str[i];
			hash *= 16777619u;
		}

		return hash;
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
		Require(frame, "Owner is set.");

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

	void Frame::setRenderTargets(gsl::span<Runtime::Texture*> colour_targets, Runtime::Texture* depth_target)
	{
		// Calculate the hash for the set of textures
		std::array<void*, 9> ptrs;
		ptrs[0] = depth_target;
		for (ptrdiff_t i = 0; i < colour_targets.size(); i++)
		{
			ptrs[i + 1] = colour_targets[i];
		}

		unsigned int hash = calculateFNV({ (char*)ptrs.data(), (char*)(ptrs.data() + 9) });

		// Find the FBO in the cache
		auto cache_entry = _fbos.find(hash);
		if (cache_entry != _fbos.end())
		{
			cache_entry->second.bind();
		}
		else
		{
			const Runtime::Texture* colours[8];
			for (ptrdiff_t i = 0; i < colour_targets.size(); i++)
			{
				colours[i] = colour_targets[i];
			}
			auto depth = depth_target;
			auto new_entry = _fbos.emplace(hash, OpenGL::Framebuffer{ colours, (size_t) colour_targets.size(), depth });
			new_entry.first->second.bind();
		}
	}

	void Frame::queueBufferForDeletion(owner_ptr<Buffer> buffer)
	{
		_bufferRecycleQueue.push_back(std::move(buffer));
	}

	GraphicsEngine::GraphicsEngine()
	{
		using Vcl::Graphics::OpenGL::GL;

		// Initialize glew
		glewExperimental = GL_TRUE;
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			/* Problem: glewInit failed, something is seriously wrong. */
			std::cout << "Error: GLEW: " << glewGetErrorString(err) << std::endl;
		}

		std::cout << "Status: Using OpenGL:   " << glGetString(GL_VERSION) << std::endl;
		std::cout << "Status:       Vendor:   " << glGetString(GL_VENDOR) << std::endl;
		std::cout << "Status:       Renderer: " << glGetString(GL_RENDERER) << std::endl;
		std::cout << "Status:       Profile:  " << GL::getProfileInfo() << std::endl;
		std::cout << "Status:       Shading:  " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
		std::cout << "Status: Using GLEW:     " << glewGetString(GLEW_VERSION) << std::endl;

		// Control V-Sync
		//wglSwapIntervalEXT(0);

		// Enable the synchronous debug output
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

		// Disable debug severity: notification
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);

		// Disable specific messages
		GLuint perf_messages_ids[] =
		{
			131154, // Pixel-path performance warning: Pixel transfer is synchronized with 3D rendering
		//	131218, // NVIDIA: "shader will be recompiled due to GL state mismatches"
		};
		glDebugMessageControl
		(
			GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE,
			sizeof(perf_messages_ids) / sizeof(GLuint), perf_messages_ids, GL_FALSE
		);

		// Register debug callback
		glDebugMessageCallback(OpenGLDebugMessageCallback, nullptr);

		// Read the device dependent constant
		_cbufferAlignment = GL::getInteger(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT);

		// Create three frames:
		// 1. where the CPU writes to
		// 2. which the GPU driver is processing
		// 3. which the GPU is processing
		_frames.resize(_numConcurrentFrames);
	}

	void GraphicsEngine::beginFrame()
	{
		// Increment the frame counter to indicate the start of the next frame
		incrFrameCounter();

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
		_currentFrame->readBackBuffer()->transfer();

		// Execute the callbacks of the read-back requests
		for (auto& callback : _readBackCallbacks)
		{
			callback.second(callback.first);
		}

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

	ref_ptr<DynamicTexture<3>> GraphicsEngine::allocatePersistentTexture(std::unique_ptr<Runtime::Texture> source)
	{
		std::array<std::unique_ptr<Runtime::Texture>, 3> textures;
		textures[0] = std::move(source);
		for (size_t i = 1; i < textures.size(); i++)
		{
			textures[i] = textures[0]->clone();
		}

		_persistentTextures.emplace_back(make_owner<DynamicTexture<3>>(std::move(textures)));
		return _persistentTextures.back();
	}

	void GraphicsEngine::deletePersistentTexture(ref_ptr<DynamicTexture<3>> tex)
	{

	}

	void GraphicsEngine::queueReadback(const Runtime::Texture& tex, std::function<void(const BufferView&)> callback)
	{
		// Index of the current frame
		int curr_frame_idx = currentFrame() % _numConcurrentFrames;

		// Fetch the staging area
		auto staging_area = _currentFrame->readBackBuffer();

		// Queue the copy command
		auto memory_range = staging_area->copyFrom(tex);

		// Queue the callback
		_readBackCallbacks.emplace_back(memory_range, std::move(callback));
	}

	void GraphicsEngine::enqueueCommand(std::function<void(void)> cmd)
	{
		std::unique_lock<std::mutex> lck{ _cmdMutex };

		_genericCmds.emplace_back(std::move(cmd));
	}

	void GraphicsEngine::resetRenderTargets()
	{

	}

	void GraphicsEngine::setRenderTargets(gsl::span<ref_ptr<Runtime::Texture>> colour_targets, ref_ptr<Runtime::Texture> depth_target)
	{
		// Index of the current frame
		int curr_frame_idx = currentFrame() % _numConcurrentFrames;

		// Fetch the frame we are using for the current rendering frame
		auto& curr_frame = _frames[curr_frame_idx];

		// Set the render targets for the current frame
		Runtime::Texture* colours[8];
		for (ptrdiff_t i = 0; i < colour_targets.size(); i++)
		{
			colours[i] = colour_targets[i].get();
		}

		curr_frame.setRenderTargets({ colours, colours + colour_targets.size() }, depth_target.get());
	}

	void GraphicsEngine::setRenderTargets(gsl::span<ref_ptr<DynamicTexture<3>>> colour_targets, ref_ptr<Runtime::Texture> depth_target)
	{
		// Index of the current frame
		int curr_frame_idx = currentFrame() % _numConcurrentFrames;

		// Fetch the frame we are using for the current rendering frame
		auto& curr_frame = _frames[curr_frame_idx];

		// Set the render targets for the current frame
		Runtime::Texture* colours[8];
		for (ptrdiff_t i = 0; i < colour_targets.size(); i++)
		{
			colours[i] = (*colour_targets[i])[curr_frame_idx];
		}

		curr_frame.setRenderTargets({ colours, colours + colour_targets.size() }, depth_target.get());
	}

	void GraphicsEngine::setRenderTargets(gsl::span<ref_ptr<Runtime::Texture>> colour_targets, ref_ptr<DynamicTexture<3>> depth_target)
	{
		// Index of the current frame
		int curr_frame_idx = currentFrame() % _numConcurrentFrames;

		// Fetch the frame we are using for the current rendering frame
		auto& curr_frame = _frames[curr_frame_idx];

		// Set the render targets for the current frame
		Runtime::Texture* colours[8];
		for (ptrdiff_t i = 0; i < colour_targets.size(); i++)
		{
			colours[i] = colour_targets[i].get();
		}
		auto depth = (*depth_target)[curr_frame_idx];

		curr_frame.setRenderTargets({ colours, colours + colour_targets.size() }, depth);
	}

	void GraphicsEngine::setRenderTargets(gsl::span<ref_ptr<DynamicTexture<3>>> colour_targets, ref_ptr<DynamicTexture<3>> depth_target)
	{
		// Index of the current frame
		int curr_frame_idx = currentFrame() % _numConcurrentFrames;

		// Fetch the frame we are using for the current rendering frame
		auto& curr_frame = _frames[curr_frame_idx];

		// Set the render targets for the current frame
		Runtime::Texture* colours[8];
		for (ptrdiff_t i = 0; i < colour_targets.size(); i++)
		{
			colours[i] = (*colour_targets[i])[curr_frame_idx];
		}
		auto depth = (*depth_target)[curr_frame_idx];

		curr_frame.setRenderTargets({ colours, colours + colour_targets.size() }, depth);
	}

	void GraphicsEngine::setConstantBuffer(int idx, BufferView view)
	{	
		auto& buffer = static_cast<const OpenGL::Buffer&>(view.owner());
		glBindBufferRange(GL_UNIFORM_BUFFER, idx, buffer.id(), view.offset(), view.size());
	}

	void GraphicsEngine::setPipelineState(ref_ptr<Runtime::PipelineState> state)
	{
		auto gl_state = static_pointer_cast<OpenGL::PipelineState>(state);

		gl_state->bind();
	}
}}}}
