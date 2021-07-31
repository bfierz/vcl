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
#include <vcl/graphics/imageprocessing/task.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/imageprocessing/imageprocessor.h>
#include <vcl/graphics/imageprocessing/link.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing
{
	void Task::initialize(const TaskDescription& desc)
	{
		// Store the description
		_desc = desc;

		// Create sufficient slots
		_inputSlots.reserve(_desc.Inputs.size());
		for (unsigned int i = 0; i < _desc.Inputs.size(); i++)
		{
		//	if (_desc.Inputs[i].Type == InputSlotType::Task)
		//	{
		//		_inputSlots.push_back(std::make_unique<FilterInputSlot>(_desc.Inputs[i].Name)));
		//	}
		//	else if (_desc.Inputs[i].Type == InputSlotType::Resource)
		//	{
		//		_inputSlots.push_back(std::make_unique<ResourceInputSlot>(_desc.Inputs[i].Name)));
		//	}
			_inputSlots.push_back(std::make_unique<InputSlot>(_desc.Inputs[i].Name, this));
		}

		_outputSlots.reserve(_desc.Outputs.size());
		for (unsigned int o = 0; o < _desc.Outputs.size(); o++)
		{
			_outputSlots.push_back(std::make_unique<OutputSlot>(_desc.Outputs[o].Name, this));
		}
	}

	unsigned int Task::nrOutputSlots() const
	{
		return (unsigned int)_outputSlots.size();
	}
	unsigned int Task::nrInputSlots() const
	{
		return (unsigned int)_inputSlots.size();
	}
	OutputSlot* Task::outputSlot(unsigned int idx)
	{
		return _outputSlots[idx].get();
	}
	InputSlot* Task::inputSlot(unsigned int idx)
	{
		return _inputSlots[idx].get();
	}

	void Task::updateResources(ImageProcessor* processor)
	{
		for (unsigned int i = 0; i < nrOutputSlots(); i++)
		{
			//unsigned int x = outputSlot(i)->x();
			//unsigned int y = outputSlot(i)->y();
			unsigned int w = outputSlot(i)->width();
			unsigned int h = outputSlot(i)->height();

			bool needs_new_image = true;

			/*if (outputSlot(i)->connections().size() > 0)
			{
				w = w > 0 ? outputSlot(i)->width()  : inputSlot(0)->width();
				h = h > 0 ? outputSlot(i)->height() : inputSlot(0)->height();

				if (outputSlot(i)->resource() && w <= outputSlot(i)->resource()->width() && h <= outputSlot(i)->resource()->height())
					needs_new_image = false;
			}
			else */if (nrInputSlots() > 0 && inputSlot(0)->resource())
			{
				w = inputSlot(0)->width();
				h = inputSlot(0)->height();

				if (outputSlot(i)->resource() && w <= (unsigned int)outputSlot(i)->resource()->width() && h <= (unsigned int)outputSlot(i)->resource()->height())
					needs_new_image = false;
			}

			// Update the existing resource
			if (needs_new_image)
			{
				auto fmt = _desc.Outputs[i].Format;
				if (fmt == SurfaceFormat::Unknown)
					fmt = SurfaceFormat::R8G8B8A8_UNORM;
				auto resource = processor->requestImage(w, h, fmt);
				outputSlot(i)->setResource(resource, w, h);
			}
		}
	}
}}}
