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
#include <vcl/graphics/opengl/algorithm/scan.h>

// VCL
#include <vcl/core/contract.h>

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
namespace
{
	const char* module = R"(

#version 430 core

#define WORKGROUP_SIZE 256

// Number of elements
uniform uint size;

// Local data
shared uint l_data[2 * WORKGROUP_SIZE];

// Local layout
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

///////////////////////////////////////////////////////////////////////////////
// Naive inclusive scan: O(N * log2(N)) operations
// Allocate 2 * 'size' local memory, initialize the first half
// with 'size' zeros avoiding if(pos >= offset) condition evaluation
// and saving instructions
uint scan1Inclusive(uint idata, uint size)
{
    uint pos = 2 * gl_LocalInvocationID.x - (gl_LocalInvocationID.x & (size - 1));
    l_data[pos] = 0;
    pos += size;
    l_data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1)
	{
		memoryBarrierShared();
		barrier();

        uint t = l_data[pos] + l_data[pos - offset];
		memoryBarrierShared();
		barrier();

        l_data[pos] = t;
    }

    return l_data[pos];
}

uint scan1Exclusive(uint idata, uint size)
{
    return scan1Inclusive(idata, size) - idata;
}

// Vector scan: the array to be scanned is stored
// in work-item private memory as uvec4
uvec4 scan4Inclusive(uvec4 data4, uint size)
{
    // Level-0 inclusive scan
    data4.y += data4.x;
    data4.z += data4.y;
    data4.w += data4.z;

    // Level-1 exclusive scan
    uint val = scan1Inclusive(data4.w, size / 4) - data4.w;

    return (data4 + uvec4(val));
}

uvec4 scan4Exclusive(uvec4 data4, uint size)
{
    return scan4Inclusive(data4, size) - data4;
}

// Main methods
#ifdef scanExclusiveLocal1

// Output data
layout(std430, binding = 0) buffer Destination
{
  uvec4 dst[];
};

// Input data
layout(std430, binding = 1) buffer Source
{
  uvec4 src[];
};

void main()
{
	// Load data
	uvec4 idata4 = src[gl_GlobalInvocationID.x];

	// Calculate exclusive scan
	uvec4 odata4 = scan4Exclusive(idata4, size);

	// Write back
	dst[gl_GlobalInvocationID.x] = odata4;
}
#elif defined scanExclusiveLocal2

// Output data
layout(std430, binding = 0) buffer Destination
{
  uint dst[];
};

// Input data
layout(std430, binding = 1) buffer Source
{
  uint src[];
};

// Intermediate work data
layout(std430, binding = 2) buffer Workspace
{
  uint buf[];
};

// Number of elements to process
uniform uint N;

// Exclusive scan of top elements of bottom-level scans (4 * WORKGROUP_SIZE)
void main()
{
    // Load top elements
    // Convert results of bottom-level scan back to inclusive
    // Skip loads and stores for inactive work-items of the work-group with highest index(pos >= N)
    uint data = 0;
    if (gl_GlobalInvocationID.x < N)
	{
		data =
			dst[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * gl_GlobalInvocationID.x] + 
			src[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * gl_GlobalInvocationID.x];
	}

    // Compute
    uint odata = scan1Exclusive(data, size);

    // Avoid out-of-bound access
    if (gl_GlobalInvocationID.x < N)
        buf[gl_GlobalInvocationID.x] = odata;
}

#elif defined uniformUpdate

// Output data
layout(std430, binding = 0) buffer Destination
{
  uvec4 dst[];
};

// Input data
layout(std430, binding = 1) buffer Workspace
{
  uint buf[];
};

// Top element to pass down
shared uint top;

// Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
void main()
{
    uvec4 data4 = dst[gl_GlobalInvocationID.x];

    if (gl_LocalInvocationID.x == 0)
        top = buf[gl_WorkGroupID.x];

    memoryBarrierShared();
	barrier();
    data4 += uvec4(top);
    dst[gl_GlobalInvocationID.x] = data4;
}
#endif
)";
}

namespace Vcl { namespace Graphics
{
	namespace
	{
		owner_ptr<Runtime::OpenGL::ShaderProgram> createKernel(const char* source, const char* header)
		{
			using namespace Vcl::Graphics::Runtime;

			// Compile the shader
			OpenGL::Shader cs(ShaderType::ComputeShader, 0, source, header);

			// Create the program descriptor
			OpenGL::ShaderProgramDescription desc;
			desc.ComputeShader = &cs;

			// Create the shader program
			return make_owner<OpenGL::ShaderProgram>(desc);
		}

		unsigned int iSnapUp(unsigned int dividend, unsigned int divisor)
		{
			return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
		}

		unsigned int factorRadix2(unsigned int& log2L, unsigned int L)
		{
			if (!L)
			{
				log2L = 0;
				return 0;
			}
			else
			{
				for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
				return L;
			}
		}
	}

	ScanExclusiveLarge::ScanExclusiveLarge(unsigned int maxElements)
	: _maxElements(maxElements)
	{
		using namespace Vcl::Graphics::Runtime;
		
		BufferDescription desc =
		{
			std::max(1u, maxElements / MaxWorkgroupInclusiveScanSize) * sizeof(unsigned int),
			Usage::Default,
			{}
		};

		_workSpace = make_owner<OpenGL::Buffer>(desc);

		// Load the sorting kernels
		_scanExclusiveLocal1Kernel = createKernel(module, "#define scanExclusiveLocal1\n");
		_scanExclusiveLocal2Kernel = createKernel(module, "#define scanExclusiveLocal2\n");
		_uniformUpdateKernel = createKernel(module, "#define uniformUpdate\n");
	}

	void ScanExclusiveLarge::operator()
	(
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> src,
		unsigned int batchSize,
		unsigned int arrayLength
	)
	{
		// Check power-of-two factorization
		unsigned int log2L;
		unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
		Check(factorizationRemainder == 1, "Is power of two");

		// Check supported size range
		Check((arrayLength >= MinLargeArraySize) && (arrayLength <= MaxLargeArraySize), "Array is within size");

		// Check total batch size limit
		Check((batchSize * arrayLength) <= MaxBatchElements, "Batch size is within range");

		scanExclusiveLocal1
		(
			dst,
			src,
			(batchSize * arrayLength) / (4 * WorkgroupSize),
			4 * WorkgroupSize
		);

		scanExclusiveLocal2
		(
			_workSpace,
			dst,
			src,
			batchSize,
			arrayLength / (4 * WorkgroupSize)
		);
		
		uniformUpdate
		(
			dst,
			_workSpace,
			(batchSize * arrayLength) / (4 * WorkgroupSize)
		);
	}

	void ScanExclusiveLarge::scanExclusiveLocal1
	(
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> src,
		unsigned int n,
		unsigned int size
	)
	{
		Require(_scanExclusiveLocal1Kernel, "Kernel is loaded.");

		// Bind the program to the pipeline
		_scanExclusiveLocal1Kernel->bind();

		// Bind the buffers and parameters
		_scanExclusiveLocal1Kernel->setBuffer("Destination", dst.get());
		_scanExclusiveLocal1Kernel->setBuffer("Source", src.get());
		_scanExclusiveLocal1Kernel->setUniform(_scanExclusiveLocal1Kernel->uniform("size"), size);

		// Execute the compute shader
		glDispatchCompute((n*size) / 4, 1, 1);
	}

	void ScanExclusiveLarge::scanExclusiveLocal2
	(
		ref_ptr<Runtime::OpenGL::Buffer> buffer,
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> src,
		unsigned int n,
		unsigned int size
	)
	{
		Require(_scanExclusiveLocal2Kernel, "Kernel is loaded.");

		// Bind the program to the pipeline
		_scanExclusiveLocal2Kernel->bind();

		// Number of elements to process
		unsigned int elements = n * size;

		// Bind the buffers and parameters
		_scanExclusiveLocal2Kernel->setBuffer("Destination", dst.get());
		_scanExclusiveLocal2Kernel->setBuffer("Source", src.get());
		_scanExclusiveLocal2Kernel->setBuffer("Workspace", buffer.get());
		_scanExclusiveLocal2Kernel->setUniform(_scanExclusiveLocal2Kernel->uniform("size"), size);
		_scanExclusiveLocal2Kernel->setUniform(_scanExclusiveLocal2Kernel->uniform("N"), elements);

		// Execute the compute shader
		glDispatchCompute(iSnapUp(elements, WorkgroupSize), 1, 1);
	}

	void ScanExclusiveLarge::uniformUpdate
	(
		ref_ptr<Runtime::OpenGL::Buffer> dst,
		ref_ptr<Runtime::OpenGL::Buffer> buffer,
		unsigned int n
	)
	{
		Require(_uniformUpdateKernel, "Kernel is loaded.");

		// Bind the program to the pipeline
		_uniformUpdateKernel->bind();

		// Bind the buffers and parameters
		_uniformUpdateKernel->setBuffer("Destination", dst.get());
		_uniformUpdateKernel->setBuffer("Workspace", buffer.get());

		// Execute the compute shader
		glDispatchCompute(n * WorkgroupSize, 1, 1);
	}
}}
