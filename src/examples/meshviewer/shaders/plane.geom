#version 430 core

// Convert input points to a set of 2 triangles
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// Output data
out VertexData
{
	vec4 Position;
	vec3 Normal;
} Out;

// Shader constants
uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

void main(void)
{
	vec4 eq = gl_in[0].gl_Position;
	vec3  N = eq.xyz;
	float d = eq.w;
	
	// Model-view matrix
	mat4 MV = ViewMatrix * ModelMatrix;

	// Transform to view-space
	p0 = MV * p0;
	p1 = MV * p1;
	p2 = MV * p2;
	p3 = MV * p3;

	// Assemble primitives
	Out.Normal = n; Out.Position = p0; gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.Normal = n; Out.Position = p1; gl_Position = ProjectionMatrix * p1; EmitVertex();
	Out.Normal = n; Out.Position = p2; gl_Position = ProjectionMatrix * p2; EmitVertex();
	EndPrimitive();
}
	