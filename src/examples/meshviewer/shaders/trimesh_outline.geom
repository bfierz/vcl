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
#version 430 core
#extension GL_GOOGLE_include_directive : enable
#extension GL_ARB_enhanced_layouts : enable

#include "3DSceneBindings.h"

// Convert input points to a set of max 3 line segments represented as quads
layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

// Input data from last stage
layout(location = 0) in VertexData
{
	ivec3 Indices0;
	ivec3 Indices1;
} In[1];

// Output data
out VertexData
{
	vec3 Colour;
} Out;

// Shader buffers
struct Vector3f
{
	float x, y, z;
};

layout (std430, binding = 0) buffer VertexPositions
{ 
	Vector3f Position[];
};
layout (std430, binding = 1) buffer VertexNormals
{ 
	Vector3f Normal[];
};
layout (std430, binding = 2) buffer VertexColours
{ 
	vec4 Colour[];
};

// Shader constants
uniform mat4 ModelMatrix;

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
bool isFrontFacing(vec4 p0, vec4 p1, vec4 p2)
{
	// Use determinant of cross-product
    return 0 < (p0.x * p1.y - p1.x * p0.y) + (p1.x * p2.y - p2.x * p1.y) + (p2.x * p0.y - p0.x * p2.y);
}

void emitEdge(vec4 p0, vec4 p1, vec2 n0, vec2 n1)
{
	// Compute the other points of the quad
	vec4 p2 = vec4(p1.xy + 0.005*p1.z*n1, p1.zw);
	vec4 p3 = vec4(p0.xy + 0.005*p0.z*n0, p0.zw);

	// Assemble primitives
	Out.Colour = vec3(1); gl_Position = p1; EmitVertex();
	Out.Colour = vec3(1); gl_Position = p2; EmitVertex();
	Out.Colour = vec3(1); gl_Position = p0; EmitVertex();
	Out.Colour = vec3(1); gl_Position = p3; EmitVertex();
	EndPrimitive();
}

void main(void)
{
	// Indices
	int i0 = In[0].Indices0.x;
	int i1 = In[0].Indices0.y;
	int i2 = In[0].Indices0.z;
	int i3 = In[0].Indices1.x;
	int i4 = In[0].Indices1.y;
	int i5 = In[0].Indices1.z;

	// Model-view-projection matrix
	mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

	// Screen-space vertices
	vec4 p0 = MVP * vec4(Position[i0].x, Position[i0].y, Position[i0].z, 1);
	vec4 p1 = MVP * vec4(Position[i1].x, Position[i1].y, Position[i1].z, 1);
	vec4 p2 = MVP * vec4(Position[i2].x, Position[i2].y, Position[i2].z, 1);
	vec4 p3 = MVP * vec4(Position[i3].x, Position[i3].y, Position[i3].z, 1);
	vec4 p4 = MVP * vec4(Position[i4].x, Position[i4].y, Position[i4].z, 1);
	vec4 p5 = MVP * vec4(Position[i5].x, Position[i5].y, Position[i5].z, 1);

	// Emit edges if main face is visible, and any other face is not
	if (isFrontFacing(p0, p2, p4))
	{
		vec4 n0 = MVP * vec4(Normal[i0].x, Normal[i0].y, Normal[i0].z, 0);
		vec4 n1 = MVP * vec4(Normal[i2].x, Normal[i2].y, Normal[i2].z, 0);
		vec4 n2 = MVP * vec4(Normal[i4].x, Normal[i4].y, Normal[i4].z, 0);

        if (!isFrontFacing(p0, p1, p2))	emitEdge(p0, p2, n0.xy, n1.xy);
        if (!isFrontFacing(p2, p3, p4)) emitEdge(p2, p4, n1.xy, n2.xy);
        if (!isFrontFacing(p0, p4, p5)) emitEdge(p4, p0, n2.xy, n0.xy);
	}
}
	