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
#include <vcl/graphics/imageprocessing/imageprocessor.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing
{
	void ImageProcessor::execute(Task* filter)
	{
		std::stack<Task*, std::vector<Task*>> queue;
		std::set<Task*> permanent;
		std::set<Task*> temporary;

		visit(filter, queue, permanent, temporary);

		while (!queue.empty())
		{
			auto curr = queue.top();
			queue.pop();

			curr->process(this);
		}
	}

	void ImageProcessor::visit
	(
		Task* task,
		std::stack<Task*, std::vector<Task*>>& queue,
		std::set<Task*>& permanent,
		std::set<Task*>& temporary
	)
	{
		VclRequire(temporary.find(task) == temporary.end(), "Graph is a DAG.");

		if (permanent.find(task) != permanent.end())
			return;

		temporary.insert(task);
		permanent.insert(task);
		queue.push(task);

		unsigned int nr_inputs = task->nrInputSlots();
		for (unsigned int in = 0; in < nr_inputs; in++)
		{
			auto slot = task->inputSlot(in);
			if (slot->source())
			{
				visit(slot->source()->task(), queue, permanent, temporary);
			}
		}

		temporary.erase(task);
	}
}}}
