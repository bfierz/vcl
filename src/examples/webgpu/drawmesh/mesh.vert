/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2022 Basil Fierz
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
#version 450 core

////////////////////////////////////////////////////////////////////////////////
// Shader Input
////////////////////////////////////////////////////////////////////////////////
layout(location = 0) in vec3 Position;
layout(location = 1) in uint ColorIdx;

////////////////////////////////////////////////////////////////////////////////
// Shader Output
////////////////////////////////////////////////////////////////////////////////
layout(location = 0) out vec3 OutPosition;
layout(location = 1) out vec4 OutColor;

////////////////////////////////////////////////////////////////////////////////
// Shader Constants
////////////////////////////////////////////////////////////////////////////////
layout(set=0, binding = 0) uniform transform
{
	mat4 ModelView;
	mat4 Proj;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
#define MAX_COLORS 10
vec3 meshletColors[MAX_COLORS] = {
  vec3(1,0,0), 
  vec3(0,1,0),
  vec3(0,0,1),
  vec3(1,1,0),
  vec3(1,0,1),
  vec3(0,1,1),
  vec3(1,0.5,0),
  vec3(0.5,1,0),
  vec3(0,0.5,1),
  vec3(1,1,1)
};

void main()
{
	vec4 posVS = ModelView * vec4(Position, 1);
	OutPosition = posVS.xyz;
	OutColor = vec4(meshletColors[ColorIdx % MAX_COLORS], 1);
	gl_Position = Proj * posVS;
}
