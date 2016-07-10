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
#include <vcl/graphics/opengl/algorithm/scan.h>

namespace Vcl { namespace Graphics
{
	/*!
	 * Note: This implementation is base on NVIDIAs OpenCL radix sort sample
	 */
	class RadixSort
	{
	public:
		RadixSort(unsigned int maxElements);
		virtual ~RadixSort() = default;

	public:
		/*!
		 * Sorts input arrays of unsigned integer keys and (optional) values
		 * 
		 * \param keys        Array of keys for data to be sorted
		 * \param numElements Number of elements to be sorted.  Must be <= 
		 *                    maxElements passed to the constructor
		 * \param keyBits     The number of bits in each key to use for ordering
		 */
		void operator()
		(
			ref_ptr<Runtime::OpenGL::Buffer> keys,
			unsigned int numElements,
			unsigned int keyBits
		);

	private:
		/*!
		 * Main key-only radix sort function.  Sorts in place in the keys and values 
		 * arrays, but uses the other device arrays as temporary storage.  All pointer 
		 * parameters are device pointers.  Uses cudppScan() for the prefix sum of
		 * radix counters.
		 */
		void radixSortKeysOnly(ref_ptr<Runtime::OpenGL::Buffer> keys, unsigned int numElements, unsigned int keyBits);

		/*!
		 * Perform one step of the radix sort. Sorts by nbits key bits per step, 
		 * starting at startbit.
		 */
		void radixSortStepKeysOnly(ref_ptr<Runtime::OpenGL::Buffer> keys, unsigned int nbits, unsigned int startbit, unsigned int numElements);

	private:
		void radixSortBlocksKeysOnlyOCL(ref_ptr<Runtime::OpenGL::Buffer> keys, unsigned int nbits, unsigned int startbit, unsigned int numElements);
		void findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements);
		void reorderDataKeysOnlyOCL(ref_ptr<Runtime::OpenGL::Buffer> keys, unsigned int startbit, unsigned int numElements);
		
	private: // Module, Kernels

		owner_ptr<Runtime::OpenGL::ShaderProgram> _radixSortBlocksKeysOnlyKernel;
		owner_ptr<Runtime::OpenGL::ShaderProgram> _findRadixOffsetsKernel;
		owner_ptr<Runtime::OpenGL::ShaderProgram> _reorderDataKeysOnlyKernel;

		//! Sub algorithm
		ScanExclusive _scan;

	private: // Buffers

		//! Warp size
		static const unsigned int WarpSize = 32;

		//! Work group size
		static const unsigned int LocalSize = 128;

		//! Maximum number of stored entries
		size_t _maxElements = 0;

		//! Memory objects for original keys and work space
		owner_ptr<Runtime::OpenGL::Buffer> _tmpKeys;

		//! Counter for each radix
		owner_ptr<Runtime::OpenGL::Buffer> _counters;

		//! Prefix sum of radix counters
		owner_ptr<Runtime::OpenGL::Buffer> _countersSum;

		//! Global offsets of each radix in each block
		owner_ptr<Runtime::OpenGL::Buffer> _blockOffsets;
	};
}}
