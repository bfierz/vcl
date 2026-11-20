/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2026 Basil Fierz
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

// Writes a gradient into a read-write texture of the bindless heap (shader model 6.6)
struct Parameters
{
	// Resource heap: output texture
	uint Output;

	// Size of the texture
	uint Width;
	uint Height;
};

ConstantBuffer<Parameters> params : register(b0);

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
	if (params.Width <= thread_id.x || params.Height <= thread_id.y)
	{
		return;
	}

	RWTexture2D<float4> output = ResourceDescriptorHeap[params.Output];
	output[thread_id.xy] = float4(thread_id.x / 255.0f, thread_id.y / 255.0f, 0, 1);
}
