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
#ifndef GLSL_SIMPLEOITBUFFER
#define GLSL_SIMPLEOITBUFFER

 // Define common locations
#define ABUFFER_CONFIG_LOC 11
#define ABUFFER_BUFFER_LOC 10

 ////////////////////////////////////////////////////////////////////////////////
 // Shader Configuration
 ////////////////////////////////////////////////////////////////////////////////
layout(early_fragment_tests) in;

////////////////////////////////////////////////////////////////////////////////
// Shader Output
////////////////////////////////////////////////////////////////////////////////

// Links to the first fragment in the linked list
layout(std430, binding = ABUFFER_BUFFER_LOC + 0) buffer IotFragmentLink
{
	int fragmentLink[];
};

// Pool with fragments. Stores colour (encoded as uint) in x and depth in y
layout(std430, binding = ABUFFER_BUFFER_LOC + 1) buffer IotFragmentPool
{
	int fragmentFirstFreeElement;
	int fragmentPool[];
};

////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////

UNIFORM_BUFFER(ABUFFER_CONFIG_LOC) ABufferConfig
{
	// Width of the ABuffer
	uint iotBufferWidth;

	// Height of the ABuffer
	uint iotBufferHeight;

	// Size of the entire fragment pool
	int fragmentPoolSize;
}

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
int allocateNewFragment(uvec2 coords)
{
	if (coords.x < iotBufferWidth && coords.y < iotBufferHeight)
	{
		int idx = atomicAdd(fragmentFirstFreeElement);
		if (idx < fragmentPoolSize)
		{
			int prev = atomicExchange(fragmentLink[coords.y * iotBufferWidth + coords.x], idx);
			fragmentPool[idx] = prev;

			return idx;
		}
	}
	
	return -1;
}

void writeFragment(int idx, vec3 colour, float depth)
{
	writeFragmentColour(vec4(colour, 1), depth);
}

void writeFragment(int idx, vec4 colour, float depth)
{
	uvec4 bytes = uvec4(clamp(colour * 255, 0, 255));
	uint bits = (bytes.a << 24) | (bytes.b << 16) | (bytes.g << 8) | (bytes.r);

	fragmentPool[idx] = uvec2(bits, floatBitsToUint(depth));
}

#endif // GLSL_SIMPLEOITBUFFER
