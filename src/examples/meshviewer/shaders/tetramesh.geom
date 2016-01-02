#version 430 core

// Convert input points to a set of 4 triangles
layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

// Input data from last stage
in VertexData
{
	ivec4 Indices;
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
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

uniform float VolumeScale = 0.8f;
uniform bool  UsePerVertexColour = false;

void main(void)
{
	float s = VolumeScale;

	vec4 p0, p1, p2, p3;
	vec4 c0, c1, c2, c3;
	vec3 n;

	ivec4 idx = In[0].Indices;
	if (UsePerVertexColour)
	{
		c0 = Colour[idx.x];
		c1 = Colour[idx.y];
		c2 = Colour[idx.z];
		c3 = Colour[idx.w];
	}
	else
	{
		c0 = gl_in[0].gl_Position;
		c1 = gl_in[0].gl_Position;
		c2 = gl_in[0].gl_Position;
		c3 = gl_in[0].gl_Position;
	}

	// Volume vertices
	vec4 x = vec4(Position[idx.x].x, Position[idx.x].y, Position[idx.x].z, 1);
	vec4 y = vec4(Position[idx.y].x, Position[idx.y].y, Position[idx.y].z, 1);
	vec4 z = vec4(Position[idx.z].x, Position[idx.z].y, Position[idx.z].z, 1);
	vec4 w = vec4(Position[idx.w].x, Position[idx.w].y, Position[idx.w].z, 1);

	// Volume Center
	vec4 c = 0.25 * (x + y + z + w);

	// Scaled volume vertices
	p0 = c + s * (x - c);
	p1 = c + s * (y - c);
	p2 = c + s * (z - c);
	p3 = c + s * (w - c);
	
	// Model-view matrix
	mat4 MV = ViewMatrix * ModelMatrix;

	// Transform to view-space
	p0 = MV * p0;
	p1 = MV * p1;
	p2 = MV * p2;
	p3 = MV * p3;

	// Assemble primitives
	n = normalize(cross(p2.xyz - p1.xyz, p3.xyz - p1.xyz));
	Out.Colour = c1; Out.Normal = n; Out.Position = p1; gl_Position = ProjectionMatrix * p1; EmitVertex();
	Out.Colour = c2; Out.Normal = n; Out.Position = p2; gl_Position = ProjectionMatrix * p2; EmitVertex();
	Out.Colour = c3; Out.Normal = n; Out.Position = p3; gl_Position = ProjectionMatrix * p3; EmitVertex();
	EndPrimitive();

	n = normalize(cross(p0.xyz - p2.xyz, p3.xyz - p2.xyz));
	Out.Colour = c2; Out.Normal = n; Out.Position = p2; gl_Position = ProjectionMatrix * p2; EmitVertex();
	Out.Colour = c0; Out.Normal = n; Out.Position = p0; gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.Colour = c3; Out.Normal = n; Out.Position = p3; gl_Position = ProjectionMatrix * p3; EmitVertex();
	EndPrimitive();

	n = normalize(cross(p0.xyz - p3.xyz, p1.xyz - p3.xyz));
	Out.Colour = c3; Out.Normal = n; Out.Position = p3; gl_Position = ProjectionMatrix * p3; EmitVertex();
	Out.Colour = c0; Out.Normal = n; Out.Position = p0; gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.Colour = c1; Out.Normal = n; Out.Position = p1; gl_Position = ProjectionMatrix * p1; EmitVertex();
	EndPrimitive();

	n = normalize(cross(p2.xyz - p0.xyz, p1.xyz - p0.xyz));
	Out.Colour = c0; Out.Normal = n; Out.Position = p0; gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.Colour = c2; Out.Normal = n; Out.Position = p2; gl_Position = ProjectionMatrix * p2; EmitVertex();
	Out.Colour = c1; Out.Normal = n; Out.Position = p1; gl_Position = ProjectionMatrix * p1; EmitVertex();
	EndPrimitive();
}
	