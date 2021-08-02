/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#include <vcl/config/cuda.h>

// C++ standard library
#include <string>
#include <unordered_map>

// VCL
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/cuda/kernel.h>
#include <vcl/compute/module.h>

namespace Vcl { namespace Compute { namespace Cuda {
	class Module : public Compute::Module
	{
	public:
		//! Constructor
		Module(CUmodule mod);

		Module(Module&&);
		Module& operator=(Module&&);

		Module(const Module&) = delete;
		Module& operator=(const Module&) = delete;

		//! Destructor
		virtual ~Module();

	public:
		static Core::owner_ptr<Module> loadFromBinary(Context* ctx, const int8_t* data, size_t size);

	public:
		inline operator CUmodule() const { return _module; }

	public:
		//! Access a kernel object through its name
		virtual ref_ptr<Compute::Kernel> kernel(const std::string& name) override;

	private:
		//! CUDA handle to a program module
		CUmodule _module;

		//! Kernels belonging to this module
		std::unordered_map<std::string, owner_ptr<Kernel>> _kernels;
	};
}}}
