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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opencl.h>

// VCL
#include <vcl/compute/opencl/context.h>
#include <vcl/compute/opencl/kernel.h>
#include <vcl/compute/buffer.h>
#include <vcl/compute/module.h>

namespace Vcl { namespace Core { namespace OpenCL {
	/*!
	 * Note: This implementation is base on NVIDIAs OpenCL radix sort sample
	 */
	class ScanExclusiveLarge
	{
	public:
		ScanExclusiveLarge(Vcl::Compute::OpenCL::Context* ctx, unsigned int maxElements);
		virtual ~ScanExclusiveLarge() = default;

	public:
		void operator()(
			ref_ptr<Compute::Buffer> dst,
			ref_ptr<Compute::Buffer> src,
			unsigned int batchSize,
			unsigned int arrayLength);

	private:
		void scanExclusiveLocal1(
			ref_ptr<Compute::Buffer> dst,
			ref_ptr<Compute::Buffer> src,
			unsigned int n,
			unsigned int size);
		void scanExclusiveLocal2(
			ref_ptr<Compute::Buffer> buffer,
			ref_ptr<Compute::Buffer> dst,
			ref_ptr<Compute::Buffer> src,
			unsigned int n,
			unsigned int size);
		void uniformUpdate(
			ref_ptr<Compute::Buffer> dst,
			ref_ptr<Compute::Buffer> buffer,
			unsigned int n);

	private: // Device context
		Vcl::Compute::OpenCL::Context* _ownerCtx;

	private: // Module, Kernels
		//! Module with the radix sort code
		ref_ptr<Compute::Module> _scanModule;

		ref_ptr<Compute::OpenCL::Kernel> _scanExclusiveLocal1Kernel = nullptr;
		ref_ptr<Compute::OpenCL::Kernel> _scanExclusiveLocal2Kernel = nullptr;
		ref_ptr<Compute::OpenCL::Kernel> _uniformUpdateKernel = nullptr;

	private: // Configurations
		const unsigned int MaxWorkgroupInclusiveScanSize = 1024;

		static const int WorkgroupSize = 256;
		static const unsigned int MaxBatchElements = 64 * 1048576;
		static const unsigned int MinShortArraySize = 4;
		static const unsigned int MaxShortArraySize = 4 * WorkgroupSize;
		static const unsigned int MinLargeArraySize = 8 * WorkgroupSize;
		static const unsigned int MaxLargeArraySize = 4 * WorkgroupSize * WorkgroupSize;

	private: // Buffers
		//! Maximum number of stored entries
		size_t _maxElements = 0;

		//! Memory objects for original keys and work space
		ref_ptr<Compute::Buffer> _workSpace;
	};
}}}
