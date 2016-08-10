#version 430 core
#extension GL_GOOGLE_include_directive : enable
#extension GL_ARB_enhanced_layouts : enable

#include "3DSceneBindings.h"

////////////////////////////////////////////////////////////////////////////////
// Shader Output
////////////////////////////////////////////////////////////////////////////////
layout(location = 0) out VertexData
{
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

// Colours of the box faces
uniform vec3 Colours[3] =
{
	vec3(1, 0, 0),
	vec3(0, 1, 0),
	vec3(0, 0, 1)
};

// Root position
uniform vec3 Origin = vec3(0, 0, 0);

// Size of a single cell
uniform float StepSize = 1.0f;

// Number of cells per size
uniform float Resolution = 10;

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

// Axis selection
int selector[6] = { 1, 2,
                    0, 2,
                    0, 1 };

void main()
{
	// Size of the entire bounding grid
	float size = Resolution * StepSize;

	// 5 points make a line-loop
	// Primitive 0: Along x-direction
	// Primitive 1: Along y-direction
	// Primitive 2: Along z-direction
	int primitiveID = gl_VertexID / 5;

	// Vertex index in the loop
	int nodeID = gl_VertexID % 5;

	// Accumulate the position of the node
	vec3 pos = Origin;
	Out.Colour = Colours[selector[2*primitiveID + 1]];
	switch (nodeID)
	{
	case 1:
		pos += size * Axis[selector[2*primitiveID + 0]];
		Out.Colour = Colours[selector[2*primitiveID + 0]];
		break;
	case 2:
		pos += size * Axis[selector[2*primitiveID + 0]];
		pos += size * Axis[selector[2*primitiveID + 1]];
		Out.Colour = Colours[selector[2*primitiveID + 1]];
		break;
	case 3:
		pos += size * Axis[selector[2*primitiveID + 1]];
		Out.Colour = Colours[selector[2*primitiveID + 0]];
		break;
	}

	// Displace the grid according to the 'depth' counted through the primitive IDs
	pos += float(gl_InstanceID) * StepSize * Axis[primitiveID];

	// Transform the point to view space
	gl_Position = ViewMatrix * ModelMatrix * vec4(pos, 1);
}
