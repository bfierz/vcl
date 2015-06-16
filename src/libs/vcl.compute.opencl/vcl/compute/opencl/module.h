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
#include <vcl/config/opencl.h>

// C++ standard library
#include <string>
#include <unordered_map>

// VCL
#include <vcl/compute/opencl/context.h>
#include <vcl/compute/opencl/kernel.h>
#include <vcl/compute/module.h>

namespace Vcl { namespace Compute { namespace OpenCL
{
	class Module : public Compute::Module
	{
	public:
		//! Constructor
		Module(cl_program mod);

		Module(Module&&);
		Module& operator =(Module&&);

		Module(const Module&) = delete;
		Module& operator =(const Module&) = delete;

		//! Destructor
		virtual ~Module();

	public:
		static Core::owner_ptr<Module> loadFromSource(Context* ctx, const char* source);

	public:
		operator cl_program () const;

	public:
		//! Access a kernel object through its name
		virtual Core::ref_ptr<Compute::Kernel> kernel(const std::string& name) override;

	private:
		//! OpenCL handle to a program module
		cl_program _module;

		//! Kernels belonging to this module
		std::unordered_map<std::string, Core::owner_ptr<Kernel>> _kernels;
	};
}}}
