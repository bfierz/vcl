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

////////////////////////////////////////////////////////////////////////////////
// Shader Configuration
////////////////////////////////////////////////////////////////////////////////

// Convert input points to a set of 4 triangles
layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

// Input data from last stage
layout(location = 0) in VertexData
{
	ivec4 Indices;

	int PrimitiveID;
} In[1];

// Output data
layout(location = 0) out VertexData
{
	// ID of the primitive
	flat int PrimitiveId;

} Out;

// Shader buffers
struct Vertex
{
	float x, y, z;
};

layout (std430) buffer VertexPositions
{ 
	Vertex Position[];
};

////////////////////////////////////////////////////////////////////////////////
// Shader constants
////////////////////////////////////////////////////////////////////////////////

uniform mat4 ModelMatrix;

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
void main(void)
{
	// Volume vertices
	ivec4 idx = In[0].Indices;
	vec4 p0 = vec4(Position[idx.x].x, Position[idx.x].y, Position[idx.x].z, 1);
	vec4 p1 = vec4(Position[idx.y].x, Position[idx.y].y, Position[idx.y].z, 1);
	vec4 p2 = vec4(Position[idx.z].x, Position[idx.z].y, Position[idx.z].z, 1);
	vec4 p3 = vec4(Position[idx.w].x, Position[idx.w].y, Position[idx.w].z, 1);

	// Model-view matrix
	mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

	// Transform to view-space
	p0 = MVP * p0;
	p1 = MVP * p1;
	p2 = MVP * p2;
	p3 = MVP * p3;

	// Set the common output
	Out.PrimitiveId = In[0].PrimitiveID;

	// Assemble primitives
	gl_Position = p1; EmitVertex();
	gl_Position = p2; EmitVertex();
	gl_Position = p3; EmitVertex();
	EndPrimitive();
	
	gl_Position = p2; EmitVertex();
	gl_Position = p0; EmitVertex();
	gl_Position = p3; EmitVertex();
	EndPrimitive();
	
	gl_Position = p3; EmitVertex();
	gl_Position = p0; EmitVertex();
	gl_Position = p1; EmitVertex();
	EndPrimitive();
	
	gl_Position = p0; EmitVertex();
	gl_Position = p2; EmitVertex();
	gl_Position = p1; EmitVertex();
	EndPrimitive();
}
	