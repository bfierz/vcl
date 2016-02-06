#version 430 core

// Convert input points to a set of 4 triangles
layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

// Input data from last stage
in VertexData
{
	ivec3 Indices;
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

layout (std430) buffer VertexPositions
{ 
	Vertex Position[];
};
layout (std430) buffer VertexColours
{ 
	vec4 Colour[];
};

// Shader constants
uniform mat4 ModelMatrix;

layout(std140) uniform PerFrameCameraData
{
	// Viewport (x, y, w, h)
	vec4 Viewport;

	// Transform from world to view space
	mat4 ViewMatrix;

	// Transform from view to screen space
	mat4 ProjectionMatrix;
};

uniform float ElementScale = 0.95f;
uniform bool  UsePerVertexColour = false;

void main(void)
{
	float s = ElementScale;

	vec4 p0, p1, p2, p3;
	vec4 c0, c1, c2, c3;
	vec3 n;

	ivec3 idx = In[0].Indices;
	if (UsePerVertexColour)
	{
		c0 = Colour[idx.x];
		c1 = Colour[idx.y];
		c2 = Colour[idx.z];
	}
	else
	{
		c0 = gl_in[0].gl_Position;
		c1 = gl_in[0].gl_Position;
		c2 = gl_in[0].gl_Position;
	}

	// Volume vertices
	vec4 x = vec4(Position[idx.x].x, Position[idx.x].y, Position[idx.x].z, 1);
	vec4 y = vec4(Position[idx.y].x, Position[idx.y].y, Position[idx.y].z, 1);
	vec4 z = vec4(Position[idx.z].x, Position[idx.z].y, Position[idx.z].z, 1);

	// Volume Center
	vec4 c = 0.333333f * (x + y + z);

	// Scaled volume vertices
	p0 = c + s * (x - c);
	p1 = c + s * (y - c);
	p2 = c + s * (z - c);
	
	// Model-view matrix
	mat4 MV = ViewMatrix * ModelMatrix;

	// Transform to view-space
	p0 = MV * p0;
	p1 = MV * p1;
	p2 = MV * p2;

	// Assemble primitives
	n = normalize(cross(p1.xyz - p0.xyz, p2.xyz - p0.xyz));
	Out.Colour = c0; Out.Normal = n; Out.Position = p0; gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.Colour = c1; Out.Normal = n; Out.Position = p1; gl_Position = ProjectionMatrix * p1; EmitVertex();
	Out.Colour = c2; Out.Normal = n; Out.Position = p2; gl_Position = ProjectionMatrix * p2; EmitVertex();
	EndPrimitive();
}
	