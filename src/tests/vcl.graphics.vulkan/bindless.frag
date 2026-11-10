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
#version 450
#extension GL_EXT_nonuniform_qualifier : require

// Indices into the bindless heaps
layout(push_constant) uniform Parameters
{
	// Resource heap: constant buffer
	uint Constants;

	// Resource heap: texture
	uint Texture;

	// Sampler heap: sampler
	uint Sampler;

	// Sample with the static sampler instead of the one from the heap
	uint UseStaticSampler;
} params;

// Resource heap: every type is an array at the same binding
layout(set = 0, binding = 0) uniform ConstantBuffer
{
	vec4 Tint;
} ConstantBuffers[];
layout(set = 0, binding = 0) uniform texture2D Textures[];

// Sampler heap
layout(set = 0, binding = 1) uniform sampler Samplers[];

// Static samplers
layout(set = 1, binding = 0) uniform sampler StaticSampler;

layout(location = 0) in vec2 TexCoord;

layout(location = 0) out vec4 Colour;

void main()
{
	vec4 texel;
	if (params.UseStaticSampler != 0)
		texel = texture(sampler2D(Textures[params.Texture], StaticSampler), TexCoord);
	else
		texel = texture(sampler2D(Textures[params.Texture], Samplers[params.Sampler]), TexCoord);

	Colour = texel * ConstantBuffers[params.Constants].Tint;
}
