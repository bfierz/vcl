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

// Samples a texture through the bindless heaps (shader model 6.6)
struct Parameters
{
	// Resource heap: constant buffer
	uint Constants;

	// Resource heap: texture
	uint Texture;

	// Sampler heap: sampler
	uint Sampler;

	// Sample with the static sampler instead of the one from the heap
	uint UseStaticSampler;
};

struct Tint
{
	float4 Value;
};

ConstantBuffer<Parameters> params : register(b0);

// Static samplers
SamplerState StaticSampler : register(s0, space1);

float4 main(float4 position : SV_Position, float2 tex_coord : TEXCOORD) : SV_Target
{
	ConstantBuffer<Tint> tint = ResourceDescriptorHeap[params.Constants];
	Texture2D<float4> tex = ResourceDescriptorHeap[params.Texture];

	float4 texel;
	if (params.UseStaticSampler != 0)
	{
		texel = tex.Sample(StaticSampler, tex_coord);
	}
	else
	{
		SamplerState heap_sampler = SamplerDescriptorHeap[params.Sampler];
		texel = tex.Sample(heap_sampler, tex_coord);
	}

	return texel * tint.Value;
}
