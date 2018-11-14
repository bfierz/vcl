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
const char* smoke_frag_shader = R"glsl(

#version 430 core
#extension GL_ARB_enhanced_layouts : enable

////////////////////////////////////////////////////////////////////////////////
// Shader Input
////////////////////////////////////////////////////////////////////////////////
layout(location = 0) in VertexData
{
	vec3 PositionMS;
	vec3 VolumeCoord;
	vec3 Colour;
} In;

////////////////////////////////////////////////////////////////////////////////
// Shader Output
////////////////////////////////////////////////////////////////////////////////
layout(location = 0) out vec4 FragColour;

////////////////////////////////////////////////////////////////////////////////
// Shader constants
////////////////////////////////////////////////////////////////////////////////

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

// Volume data
uniform sampler3D Density;

////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////
const float infinity = 1.0f / 0.0f;
const float nan = 0.0f / 0.0f;

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

// Method from Pharr, Humphrey
bool intersect
(
	vec3 box_min,
	vec3 box_max,
	vec3 ray_orig,
	vec3 ray_invdir,
	inout float tmin,
	inout float tmax
)
{
	float t0 = 0;
	float t1 = infinity;

	for (int i = 0; i < 3; ++i)
	{
		float tNear = (box_min[i] - ray_orig[i]) * ray_invdir[i];
		float tFar  = (box_max[i] - ray_orig[i]) * ray_invdir[i];

		if (tNear > tFar)
		{
			float tmp = tNear;
			tNear = tFar;
			tFar = tmp;
		}
		t0 = tNear > t0 ? tNear : t0;
		t1 = tFar < t1  ? tFar : t1;

		if (t0 > t1)
			return false;
	}

	tmin = t0;
	tmax = t1;
	return true;
}

void main(void)
{
	vec3 bb_min = vec3(0);
	vec3 bb_max = vec3(0) + Axis[0] + Axis[1] + Axis[2];

	// Compute the ray-casting vector
	vec3 ray = normalize(In.PositionMS - ViewPositionMS).xyz;

	// Step from 'VolumeCoord' in direction 'ray'
	ivec3 denTexSize = textureSize(Density, 0);
	vec3 stepSize = vec3(1) / vec3(denTexSize);
	vec3 step = ray * stepSize;

	// Maximum number of steps
	float tmin, tmax;
	if (!intersect(bb_min, bb_max, In.VolumeCoord.xyz, vec3(1.0f) / ray, tmin, tmax))
	{
		FragColour = vec4(0.0f);
		return;
	}
	int nrSteps = int(tmax * denTexSize + 0.5f);

	// Sample the volume along the ray
	float density = 0;

	// Find the max element
	vec3 currPos = In.VolumeCoord.xyz;
	for (int i = 0; i < nrSteps; i++)
	{
		float currDensity = texture(Density, currPos).x;
		density = max(density, currDensity);

		currPos += step;
	}

	FragColour = vec4(vec3(density), 1);
}
)glsl";
}
