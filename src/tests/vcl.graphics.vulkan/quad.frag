#version 440 core

layout(location = 0) in PerVertexData
{
  vec3 Colour;
} In;

layout(binding = 0) uniform Material
{
  float alpha;
};

layout(location = 0) out vec4 Colour;

void main()
{	
	Colour = vec4(In.Colour, alpha);
}
