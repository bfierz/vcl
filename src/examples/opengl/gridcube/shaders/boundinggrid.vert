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
#extension GL_ARB_enhanced_layouts : enable

#include "boundinggrid.h"

////////////////////////////////////////////////////////////////////////////////
// Shader Output
////////////////////////////////////////////////////////////////////////////////
layout(location = 0) out VertexData
{
	vec3 Colour;
} Out;

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Size of the entire bounding grid
	float size = Resolution * StepSize;

	// 4 points make a line-loop
	// Primitive 0: Along x-direction
	// Primitive 1: Along y-direction
	// Primitive 2: Along z-direction
	int primitiveID = gl_VertexID / 4;

	// Vertex index in the loop
	int nodeID = gl_VertexID % 4;

	// Select correct axis', 'axisOne' denotes the direction the loop is moving
	// along
	int axisOne =  primitiveID;
	int axisTwo = (primitiveID + 1) % 3;
	int axisTre = (primitiveID + 2) % 3;

	// Accumulate the position of the node
	vec3 pos = Origin;
	switch (nodeID)
	{
	case 0:
		Out.Colour = Colours[axisTre];
		break;
	case 1:
		pos += size * Axis[axisTwo];
		Out.Colour = Colours[axisTwo];
		break;
	case 2:
		pos += size * Axis[axisTwo];
		pos += size * Axis[axisTre];
		Out.Colour = Colours[axisTre];
		break;
	case 3:
		pos += size * Axis[axisTre];
		Out.Colour = Colours[axisTwo];
		break;
	}

	// Displace the grid according to the 'depth' counted through the primitive IDs
	pos += float(gl_InstanceID) * StepSize * Axis[axisOne];

	// Transform the point to view space
	gl_Position = ViewProjectionMatrix * ModelMatrix * vec4(pos, 1);
}
