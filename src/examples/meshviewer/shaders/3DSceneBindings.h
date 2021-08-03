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
#ifndef GLSL_3D_SCENE_BINDINGS
#define GLSL_3D_SCENE_BINDINGS

#include "UniformBuffer.h"

// Define common locations
#define PER_FRAME_CAMERA_DATA_LOC 0
#define PER_FRAME_LIGHT_DATA_LOC 1

// Define common buffers
UNIFORM_BUFFER(PER_FRAME_CAMERA_DATA_LOC)
PerFrameCameraData
{
	// Viewport (x, y, w, h)
	vec4 Viewport;

	// Frustum (tan(fov / 2), aspect_ratio, near, far)
	vec4 Frustum;

	// Transform from world to view space
	mat4 ViewMatrix;

	// Transform from view to screen space
	mat4 ProjectionMatrix;
};

struct HemisphereLight
{
	// Colour of the sky
	vec3 SkyColour;

	// Colour of the ground
	vec3 GroundColor;

	// Main direction of the light
	vec3 Direction;
};

UNIFORM_BUFFER(PER_FRAME_LIGHT_DATA_LOC)
PerFrameLightData
{
	HemisphereLight HemiLight;
};

#endif // GLSL_3D_SCENE_BINDINGS
