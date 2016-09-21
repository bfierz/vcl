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

// Convert input points to a set of 4 triangles
layout(points) in;
layout(invocations = 20) in;
layout(triangle_strip, max_vertices = 12) out;

// Input data from last stage
in VertexData
{
	ivec4 Indices;
} In[1];

// Output data
out VertexData
{
	vec4 Position;
	vec3 Normal;
	vec4 Colour;
} Out;

// Shader buffers
struct Vertex
{
	float x, y, z;
};

layout (std430, binding = 0) buffer VertexPositions
{ 
	Vertex Position[];
};

// Model to World transform
uniform mat4 ModelMatrix;

// Radius of the 'nodes'
uniform float Radius = 0.1;

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

// Geometry
#define X 0.525731112119133606f
#define Z 0.850650808352039932f
 
vec3 nodes[12] =
{
	vec3(-X, 0.0f,  Z),
	vec3( X, 0.0f,  Z),
	vec3(-X, 0.0f, -Z),
	vec3( X, 0.0f, -Z),
	vec3(0.0f,  Z,  X),
	vec3(0.0f,  Z, -X),
	vec3(0.0f, -Z,  X),
	vec3(0.0f, -Z, -X),
	vec3( Z,  X, 0.0f),
	vec3(-Z,  X, 0.0f),
	vec3( Z, -X, 0.0f),
	vec3(-Z, -X, 0.0f)
};
  
ivec3 indices[20] =
{
	ivec3( 1,  4, 0), ivec3( 4, 9, 0), ivec3(4,  5, 9), ivec3(8, 5,  4), ivec3( 1, 8, 4), 
	ivec3( 1, 10, 8), ivec3(10, 3, 8), ivec3(8,  3, 5), ivec3(3, 2,  5), ivec3( 3, 7, 2), 
	ivec3( 3, 10, 7), ivec3(10, 6, 7), ivec3(6, 11, 7), ivec3(6, 0, 11), ivec3( 6, 1, 0), 
	ivec3(10,  1, 6), ivec3(11, 0, 9), ivec3(2, 11, 9), ivec3(5, 2,  9), ivec3(11, 2, 7) 
};

void renderFace(vec3 P0, vec3 P1, vec3 P2, vec3 N0, vec3 N1, vec3 N2, vec4 colour)
{
	Out.Colour = colour;
	Out.Normal = (ViewMatrix * vec4(N0, 0)).xyz;
	Out.Position = ViewMatrix * ModelMatrix * vec4(P0, 1);
	gl_Position = ProjectionMatrix * Out.Position;
	EmitVertex();

	Out.Colour = colour;
	Out.Normal = (ViewMatrix * vec4(N1, 0)).xyz;
	Out.Position = ViewMatrix * ModelMatrix * vec4(P1, 1);
	gl_Position = ProjectionMatrix * Out.Position;
	EmitVertex();

	Out.Colour = colour;
	Out.Normal = (ViewMatrix * vec4(N2, 0)).xyz;
	Out.Position = ViewMatrix * ModelMatrix * vec4(P2, 1);
	gl_Position = ProjectionMatrix * Out.Position;
	EmitVertex();

	EndPrimitive();
}

void main()
{
	// Fetch position data
	ivec4 idx = In[0].Indices;
	
	vec3 p0 = vec3(Position[idx.x].x, Position[idx.x].y, Position[idx.x].z);
	vec3 p1 = vec3(Position[idx.y].x, Position[idx.y].y, Position[idx.y].z);
	vec3 p2 = vec3(Position[idx.z].x, Position[idx.z].y, Position[idx.z].z);
	vec3 p3 = vec3(Position[idx.w].x, Position[idx.w].y, Position[idx.w].z);
	
	ivec3 icoIdx = indices[gl_InvocationID];
	vec3  icoP0 = Radius * nodes[icoIdx.x];
	vec3  icoP1 = Radius * nodes[icoIdx.y];
	vec3  icoP2 = Radius * nodes[icoIdx.z];
	vec3  icoN0 = normalize(icoP0);
	vec3  icoN1 = normalize(icoP1);
	vec3  icoN2 = normalize(icoP2);

	vec4 colour = vec4(0.8f, 0.2f, 0.2f, 1);

	renderFace(p0 + icoP0, p0 + icoP1, p0 + icoP2, icoN0, icoN1, icoN2, colour);
	renderFace(p1 + icoP0, p1 + icoP1, p1 + icoP2, icoN0, icoN1, icoN2, colour);
	renderFace(p2 + icoP0, p2 + icoP1, p2 + icoP2, icoN0, icoN1, icoN2, colour);
	renderFace(p3 + icoP0, p3 + icoP1, p3 + icoP2, icoN0, icoN1, icoN2, colour);
}
