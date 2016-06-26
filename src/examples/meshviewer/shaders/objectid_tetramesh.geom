#version 430 core
#extension GL_ARB_enhanced_layouts : enable

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
	// IDs of the vertices that go with the barycentric coords
	flat ivec4 VertexIds;

	// Barycentric coords
	vec2 BarycentricCoords;
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

// Shader constants
uniform mat4 ModelMatrix;

layout(std140, binding = 0) uniform PerFrameCameraData
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

void main(void)
{
	// Volume vertices
	ivec4 idx = In[0].Indices;
	vec4 p0 = vec4(Position[idx.x].x, Position[idx.x].y, Position[idx.x].z, 1);
	vec4 p1 = vec4(Position[idx.y].x, Position[idx.y].y, Position[idx.y].z, 1);
	vec4 p2 = vec4(Position[idx.z].x, Position[idx.z].y, Position[idx.z].z, 1);
	vec4 p3 = vec4(Position[idx.w].x, Position[idx.w].y, Position[idx.w].z, 1);

	// Model-view matrix
	mat4 MV = ViewMatrix * ModelMatrix;

	// Transform to view-space
	p0 = MV * p0;
	p1 = MV * p1;
	p2 = MV * p2;
	p3 = MV * p3;

	// Assemble primitives
	Out.VertexIds = ivec4(idx.yzw, -1);
	Out.BarycentricCoords = vec2(0, 0); gl_Position = ProjectionMatrix * p1; EmitVertex();
	Out.BarycentricCoords = vec2(1, 0); gl_Position = ProjectionMatrix * p2; EmitVertex();
	Out.BarycentricCoords = vec2(0, 1); gl_Position = ProjectionMatrix * p3; EmitVertex();
	EndPrimitive();
	
	Out.VertexIds = ivec4(idx.zxw, -1);
	Out.BarycentricCoords = vec2(0, 0); gl_Position = ProjectionMatrix * p2; EmitVertex();
	Out.BarycentricCoords = vec2(1, 0); gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.BarycentricCoords = vec2(0, 1); gl_Position = ProjectionMatrix * p3; EmitVertex();
	EndPrimitive();
	
	Out.VertexIds = ivec4(idx.wxy, -1);
	Out.BarycentricCoords = vec2(0, 0); gl_Position = ProjectionMatrix * p3; EmitVertex();
	Out.BarycentricCoords = vec2(1, 0); gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.BarycentricCoords = vec2(0, 1); gl_Position = ProjectionMatrix * p1; EmitVertex();
	EndPrimitive();
	
	Out.VertexIds = ivec4(idx.xzy, -1);
	Out.BarycentricCoords = vec2(0, 0); gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.BarycentricCoords = vec2(1, 0); gl_Position = ProjectionMatrix * p2; EmitVertex();
	Out.BarycentricCoords = vec2(0, 1); gl_Position = ProjectionMatrix * p1; EmitVertex();
	EndPrimitive();
}
	