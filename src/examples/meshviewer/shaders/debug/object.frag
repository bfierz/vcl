#version 400 core

// Input data from last stage
in VertexData
{
	vec4 Position;
	vec3 Normal;
	vec4 Colour;
} In;

// Output data
out vec4 FragColour;

void main(void)
{
	vec3 V = normalize(-In.Position.xyz);
	vec3 N = normalize(In.Normal);

	FragColour = vec4(In.Colour.rgb * dot(N, V), 1);
}
