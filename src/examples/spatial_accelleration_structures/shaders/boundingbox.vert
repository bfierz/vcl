/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2021 Basil Fierz
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
#version 450 core
#extension GL_GOOGLE_include_directive : enable

#include "boundingbox.h"

// Positions of unit bounding box
layout(location = 0) in vec3 Position;

// Per-instance minimum of bounding box
layout(location = 1) in vec3 Min;
// Per-instance maximum of bounding box
layout(location = 2) in vec3 Max;

layout(location = 0) out VertexData
{
	// View-space position
	vec3 Position;

	// Object colour
	vec4 Colour;

} Out;

void main()
{
	vec3 scale = Max - Min;

	mat4 modelView = ViewMatrix * ModelMatrix;
	vec4 posVS = modelView * vec4(Min + scale * Position, 1);
	Out.Position = posVS.xyz;
	Out.Colour = Colour;

	gl_Position = ProjectionMatrix * posVS;
}
