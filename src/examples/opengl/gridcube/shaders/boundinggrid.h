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
#ifndef GLSL_BOUNDINGGRID_H
#define GLSL_BOUNDINGGRID_H

#include <vcl/graphics/opengl/glsl/uniformbuffer.h>

////////////////////////////////////////////////////////////////////////////////
// Shader constants
////////////////////////////////////////////////////////////////////////////////
UNIFORM_BUFFER(0)
TransformData
{
	// Transform to world space
	mat4 ModelMatrix;

	// Transform from world to normalized device coordinates
	mat4 ViewProjectionMatrix;
};

UNIFORM_BUFFER(1)
BoundingGridConfig
{
	// Axis' in model space
	vec3 Axis[3];

	// Colours of the box faces
	vec3 Colours[3];

	// Root position
	vec3_u Origin;

	// Size of a single cell
	float StepSize;

	// Number of cells per size
	float Resolution;
};

#endif // GLSL_BOUNDINGGRID_H
