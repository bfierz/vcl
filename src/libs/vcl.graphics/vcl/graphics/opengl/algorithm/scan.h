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

// VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics {
	/*!
	 * Note: This implementation is base on NVIDIAs OpenCL scan sample
	 */
	class ScanExclusive
	{
	public:
		ScanExclusive(unsigned int maxElements);
		virtual ~ScanExclusive() = default;

	public:
		void operator()
		(
			ref_ptr<Runtime::OpenGL::Buffer> dst,
			ref_ptr<Runtime::OpenGL::Buffer> src,
			unsigned int arrayLength
		);

	private:
		void scanExclusiveSmall
		(
			ref_ptr<Runtime::OpenGL::Buffer> dst,
			ref_ptr<Runtime::OpenGL::Buffer> src,
			unsigned int batchSize,
			unsigned int arrayLength
		);

		void scanExclusiveLarge
		(
			ref_ptr<Runtime::OpenGL::Buffer> dst,
			ref_ptr<Runtime::OpenGL::Buffer> src,
			unsigned int batchSize,
			unsigned int arrayLength
		);

		void scanExclusiveLocal1
		(
			ref_ptr<Runtime::OpenGL::Buffer> dst,
			ref_ptr<Runtime::OpenGL::Buffer> src,
			unsigned int n,
			unsigned int size
		);
		void scanExclusiveLocal2
		(
			ref_ptr<Runtime::OpenGL::Buffer> buffer,
			ref_ptr<Runtime::OpenGL::Buffer> dst,
			ref_ptr<Runtime::OpenGL::Buffer> src,
			unsigned int n,
			unsigned int size
		);
		void uniformUpdate
		(
			ref_ptr<Runtime::OpenGL::Buffer> dst,
			ref_ptr<Runtime::OpenGL::Buffer> buffer,
			unsigned int n
		);

	private: // Module, Kernels
		std::unique_ptr<Runtime::OpenGL::ShaderProgram> _scanExclusiveLocal1Kernel;
		std::unique_ptr<Runtime::OpenGL::ShaderProgram> _scanExclusiveLocal2Kernel;
		std::unique_ptr<Runtime::OpenGL::ShaderProgram> _uniformUpdateKernel;

	private: // Configurations
		static const unsigned int MaxWorkgroupInclusiveScanSize = 1024;

		static const unsigned int WorkgroupSize = 256;
		static const unsigned int MaxBatchElements = 64 * 1048576;
		static const unsigned int MinShortArraySize = 4;
		static const unsigned int MaxShortArraySize = 4 * WorkgroupSize;
		static const unsigned int MinLargeArraySize = 4 * WorkgroupSize;
		static const unsigned int MaxLargeArraySize = 4 * WorkgroupSize * WorkgroupSize;

	private: // Buffers
		//! Maximum number of stored entries
		size_t _maxElements = 0;

		//! Memory objects for original keys and work space
		owner_ptr<Runtime::OpenGL::Buffer> _workSpace;
	};
}}

#endif // VCL_OPENGL_SUPPORT
