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
#include <vcl/components/entitymanager.h>

namespace Vcl { namespace Components
{
	Entity EntityManager::create()
	{
		uint32_t index, generation;
		if (_freeIndices.empty())
		{
			// Get the next free index
			index = static_cast<uint32_t>(_generations.size());

			// Allocate the new index
			_generations.push_back(1);

			// Return the new generation for construction
			generation = _generations.back();
		} else
		{
			index = _freeIndices.back();
			_freeIndices.pop_back();
			generation = _generations[index];
		}

		return { this, index, generation };
	}

	void EntityManager::destroy(Entity e)
	{
		VclRequire(e._manager == this, "Entity belongs the this system.");

		uint32_t index = e._id.id();
		_generations[index]++;
		_freeIndices.push_back(index);
	}
}}
