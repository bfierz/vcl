#version 430 core
#extension GL_ARB_enhanced_layouts : enable

// Convert input points to a set of 2 triangles
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// Output data
out VertexData
{
	vec4 Position;
	vec3 Normal;
	vec4 Colour;
} Out;

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

vec4 computeFrustumSize(vec4 frustum)
{
	// tan(fov / 2)
	float scale = frustum.x;
	float ratio = frustum.y;
	float near_dist = frustum.z;
	float far_dist  = frustum.w;

	float near_half_height = scale * near_dist;
	float near_half_width  = near_half_height * ratio;
	
	float far_half_height = scale * far_dist;
	float far_half_width  = far_half_height * ratio;
	
	return vec4(near_half_width, near_half_height, far_half_width, far_half_height);
}

vec3 intersectRayPlane(vec3 p0, vec3 dir, vec4 plane)
{
	vec3  N = plane.xyz;
	float d = plane.w;

	float t = -(dot(p0, N) + d) / dot(dir, N); 
	return p0 + t*dir;
}

// Salama, Kolb - A Vertex Program for Efficient Box-Plane Intersection
// http://www.cg.informatik.uni-siegen.de/data/Publications/2005/rezksalamaVMV2005.pdf
void computePlaneBoxIntersection()
{
}

vec3 computePerpendicular(vec3 n)
{
	// Find the smallest component
	int min=0;
	for (int i=1; i<3; ++i)
		if (abs(n[min]) > abs(n[i]))
			min = i;

	// Get the other two indices
	int a = (min + 1) % 3;
	int b = (min + 2) % 3;

	vec3 result;
	result[min] =  0.0f;
	result[a]   =  n[b];
	result[b]   = -n[a];
	return result;
}

void main(void)
{
	vec4 eq = gl_in[0].gl_Position;
	vec3  N = eq.xyz;
	float d = eq.w;

	// Point on plane
	vec3 P = d * N;

	// Model-view matrix
	mat4 MV = ViewMatrix * ModelMatrix;
	
#if 0
	// Transform the plane normal to the view-space
	P = (MV * vec4(P, 1)).xyz;
	N = mat3(MV) * N;
	d = dot(P, N);

	// Compute the rays of the frustum from camera point into screen
	vec4 frustum_size = computeFrustumSize(Frustum);
	vec3 point_on_far  = vec3(0, 0, -Frustum.w);
	
	vec3 d0 = normalize(point_on_far - vec3(1, 0, 0) * frustum_size.z - vec3(0, 1, 0) * frustum_size.w);
	vec3 d1 = normalize(point_on_far + vec3(1, 0, 0) * frustum_size.z - vec3(0, 1, 0) * frustum_size.w);
	vec3 d2 = normalize(point_on_far - vec3(1, 0, 0) * frustum_size.z + vec3(0, 1, 0) * frustum_size.w);
	vec3 d3 = normalize(point_on_far + vec3(1, 0, 0) * frustum_size.z + vec3(0, 1, 0) * frustum_size.w);

	// Finally compute plane corners in view space
	vec4 p0 = vec4(intersectRayPlane(vec3(0), d0, vec4(N, d)), 1);
	vec4 p1 = vec4(intersectRayPlane(vec3(0), d1, vec4(N, d)), 1);
	vec4 p2 = vec4(intersectRayPlane(vec3(0), d2, vec4(N, d)), 1);
	vec4 p3 = vec4(intersectRayPlane(vec3(0), d3, vec4(N, d)), 1);
#else

	vec3 u = normalize(computePerpendicular(N));
	vec3 v = cross(u, N);
	
	vec4 p0 = MV * vec4(P - u * 50 - v * 50, 1);
	vec4 p1 = MV * vec4(P + u * 50 - v * 50, 1);
	vec4 p2 = MV * vec4(P - u * 50 + v * 50, 1);
	vec4 p3 = MV * vec4(P + u * 50 + v * 50, 1);
	
	N = mat3(MV) * N;

#endif

	// Assemble primitives
	Out.Colour = vec4(0.75, 0.75, 0.75, 1); Out.Normal = N; Out.Position = p0; gl_Position = ProjectionMatrix * p0; EmitVertex();
	Out.Colour = vec4(0.75, 0.75, 0.75, 1); Out.Normal = N; Out.Position = p1; gl_Position = ProjectionMatrix * p1; EmitVertex();
	Out.Colour = vec4(0.75, 0.75, 0.75, 1); Out.Normal = N; Out.Position = p2; gl_Position = ProjectionMatrix * p2; EmitVertex();
	Out.Colour = vec4(0.75, 0.75, 0.75, 1); Out.Normal = N; Out.Position = p3; gl_Position = ProjectionMatrix * p3; EmitVertex();
	EndPrimitive();
}
