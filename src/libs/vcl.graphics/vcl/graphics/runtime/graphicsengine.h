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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <memory>

// VCL
#include <vcl/core/3rdparty/gsl/span.h>
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/graphics/runtime/resource/buffer.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/runtime/dynamictexture.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	class BufferView
	{
	public:
		BufferView(ref_ptr<Buffer> buf, size_t offset, size_t size, void* base_data_ptr = nullptr)
		: _owner(buf), _offsetInBytes(offset), _sizeInBytes(size)
		{
			if (base_data_ptr)
			{
				char* ptr = reinterpret_cast<char*>(base_data_ptr);
				ptr += offset;
				_data = ptr;
			}
		}

	public:
		size_t offset() const { return _offsetInBytes; }
		size_t size() const { return _sizeInBytes; }

	public:
		void* data() { return _data; }

	public:
		const Buffer& owner() const { return *_owner; }

	protected:
		//! Buffer to which the view belongs
		ref_ptr<Buffer> _owner;

		//! Offset in bytes
		size_t _offsetInBytes{ 0 };

		//! Size in bytes
		size_t _sizeInBytes{ 0 };

		//! Pointer to mapped data
		void* _data{ nullptr };
	};

	class GraphicsEngine
	{
	public:
		//! Begin a new frame
		virtual void beginFrame() = 0;

		//! End the current frame
		virtual void endFrame() = 0;

	public: // Resource allocation

		//! Request a new constant buffer for per frame data
		virtual BufferView requestPerFrameConstantBuffer(size_t size) = 0;

		//! Convertes a regular texture to a new dynamic texture with memory for each of the parallel frames
		virtual ref_ptr<DynamicTexture<3>> allocateDynamicTexture(std::unique_ptr<Texture> tex) = 0;

		//virtual void queueReadback() = 0;

	public: // Resource management
		virtual void setRenderTargets(gsl::span<ref_ptr<DynamicTexture<3>>> colour_targets, ref_ptr<DynamicTexture<3>> depth_target) = 0;

		virtual void setConstantBuffer(int idx, BufferView buffer) = 0;

	public: // Command buffer operations
		//! Set a new pipeline state
		virtual void setPipelineState(ref_ptr<PipelineState> state) = 0;

	protected:
		uint64_t currentFrame() const { return _currentFrame; }
		void incrFrameCounter() { _currentFrame++; }

	private:
		uint64_t _currentFrame{ 0 };
	};
}}}
