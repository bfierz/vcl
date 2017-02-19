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

// Data from input-assembler stage
in vec3 Position;

// Output data
layout(location = 0) out VertexData
{
	// ID of the primitive
	flat int PrimitiveId;

} Out;

// Shader constants
uniform mat4 ModelMatrix;

// Arrow transformations
uniform mat4 Transforms[] = 
{
	// Eigen::AngleAxisf(-pi / 2.0, Eigen::Vector3f::UnitZ())
	mat4(0,-1, 0, 0,
		 1, 0, 0, 0,
		 0, 0, 1, 0,
		 0, 0, 0, 1),

	// Identity
	mat4(1, 0, 0, 0,
		 0, 1, 0, 0,
		 0, 0, 1, 0,
		 0, 0, 0, 1),

	// Eigen::AngleAxisf(pi / 2.0, Eigen::Vector3f::UnitX())
	mat4(1, 0, 0, 0,
		 0, 0, 1, 0,
		 0,-1, 0, 0,
		 0, 0, 0, 1),
};

void main()
{
	// Arrow is oriented along the y-axis, rearrange colour lookup order
	// From: 0 -> y, 1 -> z, 2 -> x
	// To: 0 -> x, 1 -> y, 2 -> z
	int instance_idx = (gl_InstanceID + 1) % 3;

	// Pass index data to next stage
	Out.PrimitiveId = 1 << instance_idx;

	vec4 pos_vs  = ViewMatrix * ModelMatrix * Transforms[instance_idx] * vec4(Position, 1);
	gl_Position = ProjectionMatrix * pos_vs;
}
