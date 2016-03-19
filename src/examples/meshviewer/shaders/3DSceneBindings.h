#ifndef GLSL_3D_SCENE_BINDINGS
#define GLSL_3D_SCENE_BINDINGS

#ifdef __cplusplus
#	define UNIFORM_BUFFER(loc) struct
	
	namespace std140
	{
		struct vec3
		{
			float x, y, z;

			vec3() {}
			vec3(const Eigen::Vector3f& v) : x(v.x()), y(v.y()), z(v.z()) {}
			
		private:
			float pad;
		};

		struct vec4
		{
			float x, y, z, w;

			vec4() {}
			vec4(const Eigen::Vector4f& v) : x(v.x()), y(v.y()), z(v.z()), w(v.w()) {}
		};

		struct mat4
		{
			vec4 cols[4];

			mat4() {}
			mat4(const Eigen::Matrix4f& m)
			{
				cols[0] = vec4{ Eigen::Vector4f{ m.col(0) } };
				cols[1] = vec4{ Eigen::Vector4f{ m.col(1) } };
				cols[2] = vec4{ Eigen::Vector4f{ m.col(2) } };
				cols[3] = vec4{ Eigen::Vector4f{ m.col(3) } };
			}
		};
	}
	
	using std140::vec3;
	using std140::vec4;
	using std140::mat4;

#else
#	define UNIFORM_BUFFER(loc) layout(std140, binding = loc) uniform
#endif // __cplusplus

// Define common locations
#define PER_FRAME_CAMERA_DATA_LOC 0
#define PER_FRAME_LIGHT_DATA_LOC 1

// Define common buffers
UNIFORM_BUFFER(PER_FRAME_CAMERA_DATA_LOC) PerFrameCameraData
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

UNIFORM_BUFFER(PER_FRAME_LIGHT_DATA_LOC) PerFrameLightData
{
	HemisphereLight HemiLight;
};

#endif // GLSL_3D_SCENE_BINDINGS
