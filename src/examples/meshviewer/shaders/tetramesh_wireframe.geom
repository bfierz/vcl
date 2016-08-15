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
layout(invocations = 6) in;
layout(triangle_strip, max_vertices = 10) out;

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

// Radius of the 'edges'
uniform float Radius = 0.05;

vec3 quatvecmul(vec4 quat, vec3 v)
{
	vec3 uv;
	uv = 2 * cross(quat.xyz, v);
	return v + quat.w * uv + cross(quat.xyz, uv);
}

vec4 computeOrientation(vec3 edge)
{
	vec3 v0 = vec3(0, 1, 0);
	vec3 v1 = normalize(edge);
	float c = dot(v0, v1);

	// if dot == 1, vectors are the same
	if (abs(c - 1) < 1e-5)
	{
		// set to identity
		return vec4(0, 0, 0, 1);
	}

	// if dot == -1, vectors are opposites
	if (abs(c + 1) < 1e-5)
	{
		return vec4(1, 0, 0, 0);
	}

	vec3 axis = cross(v0, v1);
	float s = sqrt(2*(1+c));
	float invs = 1/s;
	return vec4(axis * invs, s * 0.5);
}

/*!
 *	\param st Index of the stack to generate primitives for
 *	\param h Height of the stack element
 *	\param sl Index of the slies to generate primitives for
 *	\param slices Number of slices to generate primitives for
 */
void generateCylinder(vec4 o, vec3 c, int st, float h, int sl, int slices, vec2[4] cp, vec2[4] cn, vec4 colour)
{
	for (int j = sl; j <= sl + slices; ++j)
	{
		vec3 h0 = vec3(0, float(st + 0) * h, 0);
		vec3 h1 = vec3(0, float(st + 1) * h, 0);
		vec3 c0 = vec3(cp[j % slices].x, 0.0f, cp[j % slices].y);
		vec3 n0 = quatvecmul(o, vec3(cn[j % slices].x, 0.0f, cn[j % slices].y));

		Out.Colour = colour;
		Out.Normal = n0;
		Out.Position = ViewMatrix * ModelMatrix * vec4(quatvecmul(o, h0 + c0) + c, 1);
		gl_Position = ProjectionMatrix * Out.Position;
		EmitVertex();

		Out.Colour = colour;
		Out.Normal = n0;
		Out.Position = ViewMatrix * ModelMatrix * vec4(quatvecmul(o, h1 + c0) + c, 1);
		gl_Position = ProjectionMatrix * Out.Position;
		EmitVertex();
	}
	EndPrimitive();
}

void main(void)
{
	// Cylinder configuration
	const int stacks = 1;
	const int slices = 4;
	float radius = Radius;

	// Compute Vertices for one cylinder slide
	vec2 cp[slices];
	vec2 cn[slices];
	for (int i = 0; i < slices; i++)
	{
		float fi = i;
		float rad_xz = fi / slices * 2.0f * 3.1416;
		float sin_xz = sin(rad_xz - 3.1416);
		float cos_xz = cos(rad_xz - 3.1416);

		cp[i].x = cos_xz * radius;
		cp[i].y = sin_xz * radius;

		cn[i].x = cp[i].x;
		cn[i].y = cp[i].y;
		cn[i] = normalize(cn[i]);
	}

	ivec2 e[6] =
	{
		ivec2(1, 0),
		ivec2(2, 0),
		ivec2(3, 0),
		ivec2(2, 1),
		ivec2(3, 1),
		ivec2(3, 2)
	};

	// Fetch position data
	ivec4 idx = In[0].Indices;
	vec3 p[4];
	p[0] = vec3(Position[idx.x].x, Position[idx.x].y, Position[idx.x].z);
	p[1] = vec3(Position[idx.y].x, Position[idx.y].y, Position[idx.y].z);
	p[2] = vec3(Position[idx.z].x, Position[idx.z].y, Position[idx.z].z);
	p[3] = vec3(Position[idx.w].x, Position[idx.w].y, Position[idx.w].z);

	vec3 e0 = p[e[gl_InvocationID].x];
	vec3 e1 = p[e[gl_InvocationID].y];

	// Colour of the tube
	vec4 colour = vec4(0.5, 0.5, 0.5, 1);

	// Compute orientation (e0, e1)
	vec4 o = computeOrientation(e1 - e0);

	// Render the cylinder (e0, e1)
	generateCylinder(o, e0, 0, length(e1 - e0), 0, slices, cp, cn, colour);
}
