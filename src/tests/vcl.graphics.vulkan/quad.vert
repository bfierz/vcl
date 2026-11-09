#version 440 core

layout(location = 0) in vec2 Position;
layout(location = 1) in vec3 Colour;
layout(location = 2) in mat4 Scale;

layout(location = 0) out PerVertexData
{
  vec3 Colour;
} Out;

layout(binding = 1) uniform MatrixBlock0
{
  mat4 Modelview;
};

layout(binding = 0) uniform MatrixBlock1
{
  mat4 Projection;
};

layout(std430, binding = 2) buffer Colors
{
  vec3 Scale;
} ColorsVar[2];

void main()
{
	gl_Position = Projection*Modelview*Scale*vec4(Position, 0, 1);
	Out.Colour = ColorsVar[0].Scale*Colour;
}
