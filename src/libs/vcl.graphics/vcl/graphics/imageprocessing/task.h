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

// C++ standard library
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// VCL
#include <vcl/graphics/imageprocessing/link.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing
{
	class ImageProcessor;

	enum class InputSlotType
	{
		Filter,
		Resource
	};

	struct InputSlotDescription
	{
		std::string Name;
		InputSlotType Type;
	};

	struct OutputSlotDescription
	{
		std::string Name;
		SurfaceFormat Format;
	};

	struct TaskDescription
	{
		///! Configurations of the input slots
		std::vector<InputSlotDescription> Inputs;

		///! Configurations of the output slots
		std::vector<OutputSlotDescription> Outputs;
	};

	class Task
	{
	public: // Constructor
		Task() = default;
		virtual ~Task() = default;

	public: // Initialization
		void initialize(const TaskDescription& desc);

	public: // Access slots
		unsigned int nrOutputSlots() const;
		unsigned int nrInputSlots() const;

		OutputSlot* outputSlot(unsigned int idx);
		InputSlot* inputSlot(unsigned int idx);

	public: // Process the task
		virtual void process(ImageProcessor* processor) = 0;

	protected:
		//! Updates the used output resources
		void updateResources(ImageProcessor* processor);

	protected: // Configuration
		//! Description of the task
		TaskDescription _desc;

	protected: // Slots
		//! List of this tasks input slots
		std::vector<std::unique_ptr<InputSlot>> _inputSlots;

		//! List of this tasks output slots
		std::vector<std::unique_ptr<OutputSlot>> _outputSlots;
	};
}}}
