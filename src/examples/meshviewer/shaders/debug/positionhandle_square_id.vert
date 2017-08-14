/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#version 430 core
#extension GL_GOOGLE_include_directive : enable
#extension GL_ARB_enhanced_layouts : enable

#include "../3DSceneBindings.h"

// Output data
layout(location = 0) out VertexData
{
	// ID of the primitive
	flat int PrimitiveId;
} Out;

// Shader constants
uniform mat4 ModelMatrix;

vec3 points[] = 
{
	vec3(0, 0, 0),
	vec3(1, 0, 0),
	vec3(1, 1, 0),

	vec3(0, 0, 0),
	vec3(1, 1, 0),
	vec3(0, 1, 0),

	vec3(0, 0, 0),
	vec3(1, 0, 0),
	vec3(1, 0, 1),

	vec3(0, 0, 0),
	vec3(1, 0, 1),
	vec3(0, 0, 1),

	vec3(0, 0, 0),
	vec3(0, 1, 0),
	vec3(0, 1, 1),

	vec3(0, 0, 0),
	vec3(0, 1, 1),
	vec3(0, 0, 1),
};

void main()
{
	// Primitive 0: Along xy-plane
	// Primitive 1: Along xz-plane
	// Primitive 2: Along yz-plane
	int primitiveID = gl_VertexID / 6;

	// Pass index data to next stage
	if (primitiveID == 0)
		Out.PrimitiveId = 0x3;
	else if (primitiveID == 1)
		Out.PrimitiveId = 0x5;
	else if (primitiveID == 2)
		Out.PrimitiveId = 0x6;

	vec4 pos_vs  = ViewMatrix * ModelMatrix * vec4(0.4f * points[gl_VertexID], 1);
	gl_Position = ProjectionMatrix * pos_vs;
}
