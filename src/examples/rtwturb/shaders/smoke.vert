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
namespace
{
const char* smoke_vert_shader = R"glsl(

#version 430 core
#extension GL_ARB_enhanced_layouts : enable

////////////////////////////////////////////////////////////////////////////////
// Shader Output
////////////////////////////////////////////////////////////////////////////////
layout(location = 0) out VertexData
{
	vec3 PositionMS;
	vec3 VolumeCoord;
	vec3 Colour;
} Out;

////////////////////////////////////////////////////////////////////////////////
// Shader constants
////////////////////////////////////////////////////////////////////////////////

// Transform to world space
uniform mat4 ModelMatrix;
uniform mat4 ViewProjectionMatrix;

// Axis' in model space
uniform vec3 Axis[3] =
{
	vec3(1, 0, 0),
	vec3(0, 1, 0),
	vec3(0, 0, 1)
};

// Root position
uniform vec3 Origin = vec3(-2, -2, -2);

// Cube size
uniform vec3 GridSize = vec3(20, 20, 20);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Size of the entire cube
	float size = GridSize.x;

	// Vertex index in the loop
	int nodeID = gl_VertexID;

	// Select correct axis', 'axisOne' denotes the direction the plane is moving
	// along
	int axisOne = (gl_InstanceID + 0) % 3;
	int axisTwo = (gl_InstanceID + 1) % 3;
	int axisTre = (gl_InstanceID + 2) % 3;

	// Select the side of the axis
	int side = (gl_InstanceID < 3) ? 0 : 1;
	
	// Accumulate the position of the node
	vec3 pos = Origin;
	vec3 vpos = vec3(0);
	switch (nodeID)
	{
	case 0:
		break;
	case 1:
		vpos += Axis[axisTwo];
		pos += size * Axis[axisTwo];
		break;
	case 2:
		vpos += Axis[axisTre];
		pos += size * Axis[axisTre];
		break;
	case 3:
		vpos += Axis[axisTwo];
		vpos += Axis[axisTre];
		pos += size * Axis[axisTwo];
		pos += size * Axis[axisTre];
		break;
	}

	// Debug colour
	Out.Colour = vec3(0.7f);

	// Displace the cube plane according to the 'depth'
	pos += float(side) * size * Axis[axisOne];
	vpos += float(side) * Axis[axisOne];

	// Store the world position to compute the view-ray later
	Out.PositionMS = pos;

	// Volume texture coordinate
	Out.VolumeCoord = vpos;

	// Transform the point to view space
	gl_Position = ViewProjectionMatrix * ModelMatrix * vec4(pos, 1);
}
)glsl";
}
