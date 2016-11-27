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
const char* smoke_geom_shader = R"glsl(

#version 430 core
#extension GL_ARB_enhanced_layouts : enable

////////////////////////////////////////////////////////////////////////////////
// Shader Configuration
////////////////////////////////////////////////////////////////////////////////
layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

////////////////////////////////////////////////////////////////////////////////
// Shader Input
////////////////////////////////////////////////////////////////////////////////
layout(location = 0) in VertexData
{
	vec3 PositionMS;
	vec3 VolumeCoord;
	vec3 Colour;
} In[4];

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

// Camera position
uniform vec3 ViewPositionMS;

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
void emitVertex(int v)
{
	Out.Colour = In[v].Colour;
	Out.PositionMS = In[v].PositionMS;
	Out.VolumeCoord = In[v].VolumeCoord;
	gl_Position = gl_in[v].gl_Position;
	EmitVertex();
}

void main(void)
{
	vec3 C = Origin + 0.5f*GridSize.x*Axis[0] + 0.5f*GridSize.y*Axis[1] + 0.5f*GridSize.z*Axis[2];
	vec3 F = 0.25f * (In[0].PositionMS + In[1].PositionMS + In[2].PositionMS + In[3].PositionMS);
	vec3 N = normalize(F - C);
	vec3 V = normalize(ViewPositionMS - F);

	vec3 FN = normalize(cross(In[1].PositionMS - In[0].PositionMS, In[2].PositionMS - In[0].PositionMS));

	if (dot(V, N) > 0)
	{
		if (dot(FN, N) < 0)
		{
			emitVertex(0);
			emitVertex(2);
			emitVertex(1);
			emitVertex(3);
		}
		else
		{
			emitVertex(0);
			emitVertex(1);
			emitVertex(2);
			emitVertex(3);
		}
		EndPrimitive();
	}
}
)glsl";
}
